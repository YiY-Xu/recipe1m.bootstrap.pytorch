import sys
import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.serialization import load_lua
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options

class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = pretrainedmodels.resnet50(num_classes=1000, pretrained='imagenet')
        self.dim_out = self.resnet.last_linear.in_features
        self.resnet.last_linear = None

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ImageEmbedding(nn.Module):

    def __init__(self):
        super(ImageEmbedding, self).__init__()
        self.dim_emb = Options()['model']['network']['dim_emb']
        self.activations = Options()['model']['network']['activations'] \
            if 'activations' in Options()['model']['network'] else None
        # modules
        self.convnet = ResNet()
        self.fc = nn.Linear(self.convnet.dim_out, self.dim_emb)

    def forward(self, image):
        x = self.convnet(image['data'])
        x = self.fc(x)
        if self.activations is not None:
            for name in self.activations:                
                x = nn.functional.__dict__[name](x)
        return x


class RecipeEmbedding(nn.Module):

    def __init__(self):
        super(RecipeEmbedding, self).__init__()
        self.path_ingrs = Options()['model']['network']['path_ingrs']
        self.dim_ingr_out = Options()['model']['network']['dim_ingr_out'] # 2048
        self.dim_instr_in = Options()['model']['network']['dim_instr_in']
        self.dim_instr_out = Options()['model']['network']['dim_instr_out']
        self.with_ingrs = Options()['model']['network']['with_ingrs']
        self.with_instrs = Options()['model']['network']['with_instrs']
        self.dim_emb = Options()['model']['network']['dim_emb']

        self.activations = Options()['model']['network']['activations'] \
            if 'activations' in Options()['model']['network'] else None
        # modules
        if self.with_ingrs:
            self._make_emb_ingrs()
            self.rnn_ingrs = nn.LSTM(self.dim_ingr_in, self.dim_ingr_out,
                                     bidirectional=True, batch_first=True)
        if self.with_instrs:
            self.rnn_instrs = nn.LSTM(self.dim_instr_in, self.dim_instr_out,
                                      bidirectional=False, batch_first=True)

        if 'fusion' in Options()['model']['network']:
            self.fusion = Options()['model']['network']['fusion']['name']
            if self.fusion == 'mutan':
                self.opt_mutan = {
                    'dim_hv': 2*self.dim_ingr_out,
                    'dim_hq': self.dim_instr_out,
                    'dim_mm': Options()['model']['network']['fusion']['dim_mm'],
                    'R': Options()['model']['network']['fusion']['R'],
                    'dropout_hv': Options()['model']['network']['fusion']['dropout_hv'],
                    'dropout_hq': Options()['model']['network']['fusion']['dropout_hq']
                    #'activation_hv': 'identity'
                }
                self.mutan = MutanFusion(self.opt_mutan, visual_embedding=False, question_embedding=False)
                self.dim_recipe = Options()['model']['network']['fusion']['dim_mm']
            elif self.fusion == 'mul':
                self.dim_recipe = self.dim_instr_out
        else:
            self.fusion = 'cat'
            self.dim_recipe = 0
            if self.with_ingrs:
                self.dim_recipe += 2*self.dim_ingr_out
            if self.with_instrs:
                self.dim_recipe += self.dim_instr_out
            if self.dim_recipe == 0:
                Logger()('Ingredients or/and instructions must be embedded "--model.network.with_{ingrs,instrs} True"', Logger.ERROR)
        
        self.fc = nn.Linear(self.dim_recipe, self.dim_emb)

    def forward_ingrs_instrs(self, ingrs_out=None, instrs_out=None):
        if self.with_ingrs and self.with_instrs:
            if self.fusion == 'cat':
                fusion_out = torch.cat([ingrs_out, instrs_out], 1)
            elif self.fusion == 'mul':
                # TODO: add non linearity ?
                fusion_out = torch.mul(ingrs_out, instrs_out)
            elif self.fusion == 'mutan':
                                        #visual    #question
                fusion_out = self.mutan(ingrs_out, instrs_out)
            else:
                raise ValueError()

            x = self.fc(fusion_out)

        elif self.with_ingrs:
            x = self.fc(ingrs_out)
        elif self.with_instrs:
            x = self.fc(instrs_out)

        if self.activations is not None:
            for name in self.activations:                
                x = nn.functional.__dict__[name](x)
        return x

    def forward(self, recipe):
        if self.with_ingrs:
            ingrs_out = self.forward_ingrs(recipe['ingrs'])
        else:
            ingrs_out = None

        if self.with_instrs:
            instrs_out = self.forward_instrs(recipe['instrs'])
        else:
            instrs_out = None

        x = self.forward_ingrs_instrs(ingrs_out, instrs_out)        
        return x

    def _make_emb_ingrs(self):
        data = load_lua(self.path_ingrs)
        
        self.nb_ingrs = data[0].size(0)
        self.dim_ingr_in = data[0].size(1)
        self.emb_ingrs = nn.Embedding(self.nb_ingrs, self.dim_ingr_in)

        state_dict = {}
        state_dict['weight'] = data[0]
        self.emb_ingrs.load_state_dict(state_dict)

        # idx+1 because 0 is padding
        # data[1] contains idx_to_name_ingrs (look in datasets.Recipes(HDF5))
        #self.idx_to_name_ingrs = {idx+1:name for idx, name in enumerate(data[1])}

    # def _process_lengths(self, tensor):
    #     max_length = tensor.data.size(1)
    #     lengths = list(max_length - tensor.data.eq(0).sum(1).sequeeze())
    #     return lengths

    def _sort_by_lengths(self, ingrs, lengths):
        sorted_ids = sorted(range(len(lengths)),
                            key=lambda k: lengths[k],
                            reverse=True)
        sorted_lengths = sorted(lengths, reverse=True)
        unsorted_ids = sorted(range(len(lengths)),
                              key=lambda k: sorted_ids[k])
        sorted_ids = torch.LongTensor(sorted_ids)
        unsorted_ids = torch.LongTensor(unsorted_ids)
        if ingrs.is_cuda:
            sorted_ids = sorted_ids.cuda()
            unsorted_ids = unsorted_ids.cuda()
        ingrs = ingrs[sorted_ids]
        return ingrs, sorted_lengths, unsorted_ids

    def forward_ingrs(self, ingrs):
        # TODO: to put in dataloader
        #lengths = self._process_lengths(ingrs)
        sorted_ingrs, sorted_lengths, unsorted_ids = self._sort_by_lengths(
            ingrs['data'], ingrs['lengths'])

        emb_out = self.emb_ingrs(sorted_ingrs)
        pack_out = nn.utils.rnn.pack_padded_sequence(emb_out,
            sorted_lengths, batch_first=True)

        rnn_out, (hn, cn) = self.rnn_ingrs(pack_out)
        batch_size = hn.size(1)
        hn = hn.transpose(0,1)
        hn = hn.contiguous()
        hn = hn.view(batch_size, self.dim_ingr_out*2)
        #hn = torch.cat(hn, 2) # because bidirectional
        #hn = hn.squeeze(0)
        hn = hn[unsorted_ids]
        return hn

    def forward_instrs(self, instrs):
        # TODO: to put in dataloader
        sorted_instrs, sorted_lengths, unsorted_ids = self._sort_by_lengths(
            instrs['data'], instrs['lengths'])
        pack_out = nn.utils.rnn.pack_padded_sequence(sorted_instrs,
            sorted_lengths, batch_first=True)

        rnn_out, (hn, cn) = self.rnn_instrs(sorted_instrs)
        hn = hn.squeeze(0)
        hn = hn[unsorted_ids]
        return hn

    def forward_one_ingr(self, ingrs, emb_instrs=None):
        emb_ingr = self.forward_ingrs(ingrs)
        if emb_instrs is None:
            emb_instrs = torch.zeros(1,self.dim_instr_out)
        if emb_ingr.data.is_cuda:
            emb_instrs = emb_instrs.cuda()
        #emb_instrs = Variable(emb_instrs, requires_grad=False)

        fusion_out = torch.cat([emb_ingr, emb_instrs], 1)
        x = self.fc(fusion_out)

        if self.activations is not None:
            for name in self.activations:                
                x = nn.functional.__dict__[name](x)

        return x


class Trijoint(nn.Module):

    def __init__(self):
        super(Trijoint, self).__init__()
        self.dim_emb = Options()['model']['network']['dim_emb']
        self.nb_classes = Options()['dataset']['nb_classes']
        self.with_classif = Options()['model']['with_classif']
        # modules
        self.image_embedding = ImageEmbedding()
        self.recipe_embedding = RecipeEmbedding()

        if self.with_classif:
            self.linear_classif = nn.Linear(self.dim_emb, self.nb_classes)

    def get_parameters_recipe(self):
        params = []
        params.append({'params': self.recipe_embedding.parameters()})
        if self.with_classif:
            params.append({'params': self.linear_classif.parameters()})
        params.append({'params': self.image_embedding.fc.parameters()})
        return params

    def get_parameters_image(self):
        return self.image_embedding.convnet.parameters()

    def forward(self, batch):
        out = {}
        out['image_embedding'] = self.image_embedding(batch['image'])
        out['recipe_embedding'] = self.recipe_embedding(batch['recipe'])

        if self.with_classif:
            out['image_classif'] = self.linear_classif(out['image_embedding'])
            out['recipe_classif'] = self.linear_classif(out['recipe_embedding'])

        return out
        

class AbstractFusion(nn.Module):

    def __init__(self, opt={}):
        super(AbstractFusion, self).__init__()
        self.opt = opt

    def forward(self, input_v, input_q):
        raise NotImplementedError


class MutanFusion(AbstractFusion):

    def __init__(self, opt, visual_embedding=True, question_embedding=True):
        super(MutanFusion, self).__init__(opt)
        self.visual_embedding = visual_embedding
        self.question_embedding = question_embedding
        # Modules
        if self.visual_embedding:
            self.linear_v = nn.Linear(self.opt['dim_v'], self.opt['dim_hv'])
        else:
            print('Warning fusion.py: no visual embedding before fusion')

        if self.question_embedding:
            self.linear_q = nn.Linear(self.opt['dim_q'], self.opt['dim_hq'])
        else:
            print('Warning fusion.py: no question embedding before fusion')
        
        self.list_linear_hv = nn.ModuleList([
            nn.Linear(self.opt['dim_hv'], self.opt['dim_mm'])
            for i in range(self.opt['R'])])

        self.list_linear_hq = nn.ModuleList([
            nn.Linear(self.opt['dim_hq'], self.opt['dim_mm'])
            for i in range(self.opt['R'])])

    def forward(self, input_v, input_q):
        if input_v.dim() != input_q.dim() and input_v.dim() != 2:
            raise ValueError
        batch_size = input_v.size(0)

        if self.visual_embedding:
            x_v = F.dropout(input_v, p=self.opt['dropout_v'], training=self.training)
            x_v = self.linear_v(x_v)
            if 'activation_v' in self.opt:
                    x_v = getattr(F, self.opt['activation_v'])(x_v)
        else:
            x_v = input_v

        if self.question_embedding:
            x_q = F.dropout(input_q, p=self.opt['dropout_q'], training=self.training)
            x_q = self.linear_q(x_q)
            if 'activation_q' in self.opt:
                    x_q = getattr(F, self.opt['activation_q'])(x_q)
        else:
            x_q = input_q

        x_mm = []
        for i in range(self.opt['R']):

            x_hv = F.dropout(x_v, p=self.opt['dropout_hv'], training=self.training)
            x_hv = self.list_linear_hv[i](x_hv)
            if 'activation_hv' in self.opt:
                x_hv = getattr(F, self.opt['activation_hv'])(x_hv)

            x_hq = F.dropout(x_q, p=self.opt['dropout_hq'], training=self.training)
            x_hq = self.list_linear_hq[i](x_hq)
            if 'activation_hq' in self.opt:
                x_hq = getattr(F, self.opt['activation_hq'])(x_hq)

            x_mm.append(torch.mul(x_hq, x_hv))

        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.opt['dim_mm'])

        if 'activation_mm' in self.opt:
            x_mm = getattr(F, self.opt['activation_mm'])(x_mm)

        return x_mm


# python -m im2recipe.models.trijoint
# if __name__ == '__main__':

#     bsize = 3
#     nb_ingrs_max = 7
#     nb_instrs_max = 14
#     nb_classes = 100
#     classes = ['lol']*nb_classes

#     options = {
#         'path_ingrs': '/local/cadene/data/im2recipe/text/vocab.t7',
#         'dim_image_out': 2048,
#         'dim_ingr_out': 300, # irnnDim
#         'dim_instr_in': 1024, # stDim
#         'dim_instr_out': 1024, # srnnDim
#         'dim_emb': 1024 # embDIm
#     }

#     batch = {
#         'image': {
#             'data': Variable(torch.randn(bsize, 3, 224, 224).cuda())
#             #'class_id': Variable(torch.Float)
#         },
#         'recipe': {
#             'ingrs': {
#                 'data': Variable(torch.ones(bsize, nb_ingrs_max).long().cuda()),
#                 # Variable(torch.multinomial(
#                 #             torch.Tensor([10,1,2,3]),
#                 #             bsize*nb_ingrs_max,
#                 #             replacement=True).view(bsize, -1).cuda()),
#                 'lengths': [nb_ingrs_max]*bsize#[nb_ingrs_max-1]*(bsize-3) + [3,2,1]
#             },
#             'instrs': {
#                 'data': Variable(torch.ones(bsize, nb_instrs_max, options['dim_instr_in']).cuda()),
#                 # Variable(torch.randn(bsize, nb_instrs_max).cuda()
#                 #              * torch.multinomial(
#                 #                    torch.Tensor([1,2]),
#                 #                    bsize*nb_instrs_max,
#                 #                    replacement=True).view(bsize, -1).float().cuda()),
#                 'lengths': [nb_instrs_max]*bsize#(bsize-3) + [3,2,1]
#             }
#         }
#     }

#     model = factory(options, classes=classes, cuda=True, data_parallel=False)
#     model_out = model.forward(batch)
#     print(model_out)