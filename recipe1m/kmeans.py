import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from datasets.recipe1m import Images
from sklearn.cluster import MiniBatchKMeans

def collate(batches):
    images = []
    for i in range(len(batches)):
        print(batches[i]['data'])
        img = cv2.imread(batches[i]['data'])
        blue = cv2.calcHist([img],[0],None,[256],[0,256])[1:] 
        green = cv2.calcHist([img],[1],None,[256],[0,256])[1:]
        red = cv2.calcHist([img],[2],None,[256],[0,256])[1:]
        img = np.concatenate([red, green, blue], axis=0)
        images.append(new)
    return np.array(imgs)

if __name__ == '__main__':
    image_dataset = Images('/home/ubuntu/moochi/recipe1m.bootstrap.pytorch/data/recipe1m/data_lmdb', 'val', 100, 4)
    image_dataset.items_tf = collate
    kmeans = MiniBatchKMeans(n_clusters=30, random_state=0, batch_size=100)
    for i, sample_batched in enumerate(tqdm(image_dataset.make_batch_loader(True))):
        kmeans = kmeans.partial_fit(sample_batched)

    return kmeans
