import copy
import torch
import numpy as np
import random
from torch.utils.data.sampler import Sampler
#from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.sampler import RandomSampler
#from torch.utils.data.sampler import SubsetRandomSampler
#from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.sampler import BatchSampler

from bootstrap.lib.options import Options

class RandomSamplerValues(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from

    Example:
        >>> list(RandomSamplerValues(range(10,20)))
        [15, 16, 10, 17, 11, 12, 14, 13, 18, 19]

        >>> list(RandomSampler(range(10,20)))
        [0, 4, 9, 2, 6, 1, 3, 5, 7, 8]
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        if Options()['dataset'].get("debug", False):
            generator = iter(list(range(len(self.data_source))))
        else:
            generator = iter(torch.randperm(len(self.data_source)).long())
        
        for value in generator:
            yield self.data_source[value]

    def __len__(self):
        return len(self.data_source)

class BatchSamplerCluster:
    """Sample from same clusters but not same classes"""
    def __init__(self, indices_by_cluster, batch_size, nb_indices_same_cluster):
        if batch_size % nb_indices_same_cluster != 0:
            raise ValueError('batch_size of BatchSamplerCluster ({}) must be divisible by nb_indices_same_cluster ({})'.format(
                batch_size, nb_indices_same_cluster))

        self.indices_by_cluster = indices_by_cluster
        self.batch_size = batch_size
        self.nb_indices_same_cluster = nb_indices_same_cluster

        self.batch_sampler_by_cluster = []
        for indices in indices_by_cluster:
            self.batch_sampler_by_cluster.append(
                BatchSampler(RandomSamplerValues(indices),
                             self.nb_indices_same_cluster,
                             True))

    def _make_nb_samples_by_cluster(self):
        return [len(sampler) for sampler in self.batch_sampler_by_cluster]

    def __iter__(self):
        nb_samples_by_cluster = torch.Tensor(self._make_nb_samples_by_cluster())
        gen_by_cluster = [sampler.__iter__() for sampler in self.batch_sampler_by_cluster]

        for i in range(len(self)):
            batch = []
            nb_samples = self.batch_size // self.nb_indices_same_cluster
            for j in range(nb_samples):
                # Class sampling
                if Options()['dataset'].get("debug", False):
                    idx = np.random.multinomial(1,(nb_samples_by_cluster / sum(nb_samples_by_cluster)).numpy()).argmax()
                else:
                    idx = torch.multinomial(nb_samples_by_cluster,
                        1, # num_samples
                        False)[0] #replacement

                nb_samples_by_cluster[idx] -= 1
                batch += gen_by_cluster[idx].__next__()
            yield batch

    def __len__(self):
        nb_possible_indices = sum(self._make_nb_samples_by_cluster()) * self.nb_indices_same_cluster
        return nb_possible_indices // self.batch_size

class BatchSamplerClassif(object):
    """Randomly samples indices from a list of indices grouped by class, without replacement.
    BatchSamplerClassif wraps a list of BatchSampler, one for each class (besides background).

    Arguments:
        indices_by_class (list of list of int): 
        batch_size (int): nb of indices in a batch returned by the BatchSampler
        nb_indices_same_class (int): nb of indices from the same class returned after a class sampling

    Example:
        >>> list(BatchSamplerClassif([list(range(3)), list(range(10,14)), list(range(20,25))], 4, 2))
        [[10, 13, 20, 22], [0, 1, 21, 24]]
    """

    def __init__(self, indices_by_class, batch_size, nb_indices_same_class):
        if batch_size % nb_indices_same_class != 0:
            raise ValueError('batch_size of BatchSamplerClassif ({}) must be divisible by nb_indices_same_class ({})'.format(
                batch_size, nb_indices_same_class))

        self.indices_by_class = indices_by_class
        self.batch_size = batch_size
        self.nb_indices_same_class = nb_indices_same_class

        self.batch_sampler_by_class = []
        for indices in indices_by_class:
            self.batch_sampler_by_class.append(
                BatchSampler(RandomSamplerValues(indices),
                             self.nb_indices_same_class,
                             True))

    def _make_nb_samples_by_class(self):
        """ Note that nb_samples != nb_indices
        In fact, if nb_indices = 9 and nb_indices_same_class = 2,
        then nb_samples = 4
        """
        return [len(sampler) for sampler in self.batch_sampler_by_class]
        
    def __iter__(self):
        
        nb_samples_by_class = torch.Tensor(self._make_nb_samples_by_class())
        gen_by_class = [sampler.__iter__() for sampler in self.batch_sampler_by_class]

        for i in range(len(self)):
            batch = []
            nb_samples = self.batch_size // self.nb_indices_same_class
            for j in range(nb_samples):
                # Class sampling
                if Options()['dataset'].get("debug", False):
                    idx = np.random.multinomial(1,(nb_samples_by_class / sum(nb_samples_by_class)).numpy()).argmax()
                else:
                    idx = torch.multinomial(nb_samples_by_class,
                        1, # num_samples
                        False)[0] #replacement

                nb_samples_by_class[idx] -= 1
                batch += gen_by_class[idx].__next__()
            yield batch

    def __len__(self):
        """
        Count the real number of indices using nb_samples
        
        Note that the "false" number of indices is:
            >>> nb_total_indices = sum([len(indices) for indices in self.indices_by_class])
        """
        nb_possible_indices = sum(self._make_nb_samples_by_class()) * self.nb_indices_same_class
        return nb_possible_indices // self.batch_size


class BatchSamplerClassWithNegInSameCluster(object):

    def __init__(self, indices_by_class, indices_by_cluster, cluster_by_index, batch_size, nb_indices_same_class):
        if batch_size % nb_indices_same_class != 0:
            raise ValueError('batch_size of BatchSamplerClassWithNegInSameCluster ({}) must be divisible by nb_indices_same_class ({})'.format(
                batch_size, nb_indices_same_class))
        if nb_indices_same_class % 2 != 0:
            raise ValueError('nb_indices_same_class should be an even number')

        self.indices_by_class = indices_by_class
        self.indices_by_cluster = indices_by_cluster
        self.batch_size = batch_size
        self.nb_indices_same_class = nb_indices_same_class
        self.cluster_by_index = cluster_by_index

        self.batch_sampler_by_class = []
        for indices in self.indices_by_class:
            self.batch_sampler_by_class.append(
                BatchSampler(RandomSamplerValues(indices),
                             self.nb_indices_same_class,
                             True))
        print("length of indices by class samplers is {}".format(len(indices_by_class)))

        self.batch_sampler_by_cluster = []
        for indices in self.indices_by_cluster:
            self.batch_sampler_by_cluster.append(
                BatchSampler(RandomSamplerValues(indices),
                             self.nb_indices_same_class//2,
                             True))

    def _make_nb_samples_by_class(self):
        """ Note that nb_samples != nb_indices
        In fact, if nb_indices = 9 and nb_indices_same_class = 2,
        then nb_samples = 4
        """
        return [len(sampler) for sampler in self.batch_sampler_by_class]

    def _find_cluster_index(self, samples):
        print("class id is {}".format(samples))
        clusters = []
        for sample in samples:
            for i in range(len(self.indices_by_cluster)):
                #print(sample, self.indices_by_cluster[i])
                if sample in self.indices_by_cluster[i]:
                    clusters += self.indices_by_cluster[i]
        #print("{} has no cluster".format(samples))
        return clusters
        
    def __iter__(self):
        
        nb_samples_by_class = torch.Tensor(self._make_nb_samples_by_class())
        gen_by_class = [sampler.__iter__() for sampler in self.batch_sampler_by_class]

        ### to remove
        gen_by_cluster = [sampler.__iter__() for sampler in self.batch_sampler_by_cluster]

        print("key place is here")
        print("len is {}".format(len(gen_by_class)))
        print("nb_samples_by_class: {}".format(nb_samples_by_class))
        print("nb_samples_by_class len is {}".format(len(nb_samples_by_class)))
        print("total batch is {}".format(len(self)))

        for i in range(len(self)):
            pos_batch = []
            neg_batch = []
            nb_samples = self.batch_size // self.nb_indices_same_class
            for j in range(nb_samples):
                # Class sampling
                idx = -1
                if Options()['dataset'].get("debug", False):
                    idx = np.random.multinomial(1,(nb_samples_by_class / sum(nb_samples_by_class)).numpy()).argmax()
                else:
                    idx = torch.multinomial(nb_samples_by_class,
                        1, # num_samples
                        False)[0] #replacement

                #print("yeild class id is {}".format(str(idx)))

                nb_samples_by_class[idx] -= 1
                ## generate class:
                pos_sample = gen_by_class[idx].__next__()
                #print("pos_sample is {}".format(str(pos_sample)))
                pos_batch += pos_sample

                #to remove
                cluster = self.cluster_by_index[idx]
                print("Check here cluster : {}".format(cluster))

                neg_batch += gen_by_cluster[cluster].__next__()
                print("Check here neg_batch : {}".format(len(neg_batch)))

            yield pos_batch + neg_batch

    def __len__(self):
        """
        Count the real number of indices using nb_samples
        
        Note that the "false" number of indices is:
            >>> nb_total_indices = sum([len(indices) for indices in self.indices_by_class])
        """
        nb_possible_indices = sum(self._make_nb_samples_by_class()) * self.nb_indices_same_class
        return nb_possible_indices // self.batch_size

class BatchSamplerTripletClassif(object):
    """Wraps BatchSampler for items associated to background and BatchSamplerClassif for items with classes.

    Args:
        indices_by_class (list of list of int): 
        batch_size (int): Size of mini-batch.
        pc_noclassif (float): Percentage of items associated to background in the batch
        nb_indices_same_class (int): nb of indices from the same class returned after a class sampling

    Warning: `indices_by_class` assumes that the list in position 0 contains 
             indices associated to items without classes (background)

    Warning: `pc_noclassif` is used to calculate the number of items associated to classes in the batch,
             the latter must be a multiple of `nb_indices_same_class`.

    Example:
        >>> list(BatchSamplerTripletClassif([
                list(range(8)), # indices of background
                list(range(10,14)), # class 1
                list(range(20,25)), # class 2
                list(range(30,36))], # class 3
                4, # batch_size
                pc_noclassif=0.5,
                nb_indices_same_class=2))
        [[13, 12, 2, 5], [31, 32, 4, 0], [33, 30, 6, 3], [23, 22, 7, 1]]
    """

    def __init__(self, indices_by_class, indices_by_cluster, cluster_by_index, batch_size, use_cluster, pc_noclassif=0.25, nb_indices_same_class=2, nb_indices_same_cluster=2):
        self.indices_by_cluster = copy.copy(indices_by_cluster)
        self.indices_by_class = copy.copy(indices_by_class)
        self.indices_no_class = self.indices_by_class.pop(0)
        self.cluster_by_index = cluster_by_index

        remove_target = copy.copy(self.indices_no_class)
        for index in remove_target:
            cluster = self.cluster_by_index[index]
            self.indices_by_cluster[cluster].remove(index)

        self.batch_size = batch_size
        self.pc_noclassif = pc_noclassif
        self.use_cluster = use_cluster
        self.nb_indices_same_class = nb_indices_same_class
        self.nb_indices_same_cluster = nb_indices_same_cluster

        self.batch_size_classif = round((1 - self.pc_noclassif) * self.batch_size)
        self.batch_size_noclassif = self.batch_size - self.batch_size_classif

        # Batch Sampler Same Clusters
        # self.batch_sampler_noclassif = BatchSamplerCluster(
        #     RandomSamplerValues(self.indices_by_cluster),
        #     self.batch_size_noclassif,
        #     self.nb_indices_same_cluster) if self.use_cluster else 

        self.batch_sampler_noclassif = BatchSampler(
            RandomSamplerValues(self.indices_no_class),
            self.batch_size_noclassif, True)

        # Batch Sampler Classif
        if use_cluster:
            self.batch_sampler_classif = BatchSamplerClassWithNegInSameCluster(
                RandomSamplerValues(self.indices_by_class),
                self.indices_by_cluster, #list of 30 clusters with list of index
                self.cluster_by_index,
                self.batch_size_classif,
                self.nb_indices_same_class)
        else:
            self.batch_sampler_classif = BatchSamplerClassif(
                RandomSamplerValues(self.indices_by_class),
                self.batch_size_classif,
                self.nb_indices_same_class)

        

    def __iter__(self):
        gen_classif = self.batch_sampler_classif.__iter__()
        #gen_noclassif = self.batch_sampler_noclassif.__iter__()
        gen_noclassif = self.batch_sampler_noclassif.__iter__()
        for i in range(len(self)):
            batch = []
            batch += gen_classif.__next__()
            batch += gen_noclassif.__next__()
            yield batch

    def __len__(self):
        length = min([len(self.batch_sampler_classif), len(self.batch_sampler_noclassif)])
        return length


if __name__ == '__main__':

    batch_sampler = BatchSamplerClassif([
        list(range(7)), # indices of class 1
        list(range(10,14)), # class 2
        list(range(20,25))], # class 3
        6, # batch_size
        2) # nb_indices_same_class
    print(list(batch_sampler))

    batch_sampler = BatchSamplerTripletClassif([
        list(range(9)), # indices of background
        list(range(10,14)), # class 1
        list(range(20,25)), # class 2
        list(range(30,36))], # class 3
        4, # batch_size
        pc_noclassif=0.5,
        nb_indices_same_class=2)
    #print(list(batch_sampler))
