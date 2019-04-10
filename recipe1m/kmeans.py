import os
import cv2
import pickle
import numpy as np
import lmdb
from tqdm import tqdm
from datasets.recipe1m import Images
from sklearn.cluster import MiniBatchKMeans

class Kmeans:

    def __init__(self, data_path, dir_lmdb):
        self.data_path = data_path
        self.dir_lmdb = dir_lmdb

    def get_color_histogram(self, batches):
        images = []
        for i in range(batches.shape[0]):
            img = batches[i]
            R = cv2.calcHist([img],[0],None,[256],[0,256])[1:] 
            G = cv2.calcHist([img],[1],None,[256],[0,256])[1:]
            B = cv2.calcHist([img],[2],None,[256],[0,256])[1:]
            img = np.concatenate([R, G, B], axis=0)
            images.append(img)
        return np.array(images).squeeze()

    def print_data(self, split):
        image_dataset = Images(self.data_path, split, 100, 4)
        print(image_dataset.__len__())
        for i, sample_batch in enumerate(tqdm(image_dataset.make_batch_loader(False))):
            imgs = sample_batch['data'].numpy()
            clusters = sample_batch['cluster']
            print(clusters)

    def apply_kmeans(self, split):
        with open('../kmean_models/kmean_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)

        image_dataset = Images(self.data_path, split, 100, 4)
        for i, sample_batch in enumerate(tqdm(image_dataset.make_batch_loader(False))):
            imgs = sample_batch['data'].numpy()
            indices = sample_batch['id']
            imgs = self.get_color_histogram(imgs)
            clusters = kmeans.predict(imgs)
            self.write_lmdb(split, indices, clusters)

    def apply_kmeans_leftover(self, split, start, length):
        print('here')
        with open('../kmean_models/kmean_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)

        indices = range(start, start+length)

        image_dataset = Images(self.data_path, split, 100, 4)
        for i in indices:
            print(i)
            image = image_dataset.get_image(i)
            indices = [image['id']]
            img = np.array([image['data'].numpy()])
            imgs = self.get_color_histogram(img).reshape(1, -1)
            clusters = kmeans.predict(imgs)
            #print(indices, clusters)
            self.write_lmdb(split, indices, clusters)

    def write_lmdb(self, split, indices, values):
        lmdb_env = lmdb.open(os.path.join(self.dir_lmdb, split, 'kmeans.lmdb'), map_size=int(1e9))

        for index, value in zip(indices, values):
            with lmdb_env.begin(write=True) as lmdb_txn:
                lmdb_txn.put(self.encode(index), self.encode(value))

    # Loading validation set to train the kmeans
    def train_kmeans(self):
        image_dataset = Images(self.data_path, 'val', 100, 4)
        kmeans = MiniBatchKMeans(n_clusters=30, random_state=0, batch_size=100)
        for i, sample_batch in enumerate(tqdm(image_dataset.make_batch_loader(True))):
            imgs = sample_batch['data'].numpy()
            imgs = get_color_histogram(imgs)
            kmeans = kmeans.partial_fit(imgs)

        with open('../kmean_models/kmean_model.pkl', 'wb') as f:
            pickle.dump(kmeans, f)

    def encode(self, value):
        return pickle.dumps(value)

if __name__ == '__main__':
    kmeans = Kmeans('/home/ubuntu/moochi/recipe1m.bootstrap.pytorch/data/recipe1m', 
        '/home/ubuntu/moochi/recipe1m.bootstrap.pytorch/data/recipe1m/data_lmdb')
    #kmeans.apply_kmeans('test')
    #kmeans.apply_kmeans_leftover('val', 51100, 19)
    #kmeans.print_data('val')

