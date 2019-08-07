import os, h5py
import numpy as np
import numpy.random as npr
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from functools import reduce
from joblib import dump, load

maxshift = 10000
X = np.array([])

if not os.path.exists('pca_model.joblib'):
    print("Round1")
    for shift in list(range(-maxshift, maxshift + 1, 200)):
        print(shift)
        preds = h5py.File("embedded_vals/human_hg19_promoters_1nt_FantomCorrected.bed.shift_SHIFT.h5".replace(
            'SHIFT', str(shift)), 'r')['/pred']
        X = np.concatenate([X,preds]) if X.size else preds
        print(X.shape)

    print("Fitting PCA")
    pca = PCA(0.99, whiten=True) #Incremental, batch_size=10
    X = pca.fit_transform(X)
    print(X.shape)
    dump(pca, 'pca_model.joblib')
else:
    pca = load('pca_model.joblib')

maxshift = 100000
X = np.array([])
print("Round2")
for shift in list(range(-maxshift, maxshift + 1, 200)):
    print(shift)
    preds = h5py.File("embedded_vals/human_hg19_promoters_1nt_FantomCorrected.bed.shift_SHIFT.h5".replace(
        'SHIFT', str(shift)), 'r')['/pred']
    preds = pca.transform(preds)
    preds = np.expand_dims(preds,axis=1)
    X = np.hstack([X,preds]) if X.size else preds
    print(X.shape)

compress_args = {'compression': 'gzip', 'compression_opts': 1}
out_dir="prepared_data"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
outfile = os.path.join(out_dir, 'expr_preds.h5')

genes = pd.read_csv("human_hg19_promoters_1nt_FantomCorrected.bed",
                    sep="\t", header=None, names=['chr', 'start', 'end', '.', 'strand', '?', '??', '???', 'name'])
X = X[~genes.chr.isin(['chrY']),:,:]
genes = genes[~genes.chr.isin(['chrY'])] #'chrX',
y = pd.read_csv("57epigenomes.RPKM.pc", sep="\t", index_col=0)
maskedIDs = pd.read_table("mask_histone_genes.txt", header=None) #mask histone genes, chrY genes already filtered out
y = y[y.index.isin(genes['name']) & ~y.index.isin(maskedIDs[0])] #remove rows corresponding to chrY or histone sequences
X = X[genes['name'].isin(y.index),:,:]
genes = genes[genes['name'].isin(y.index)]
y = reduce(pd.DataFrame.append, map(lambda i: y[y.index == i], genes['name']))

# print X.shape, y.shape, genes.shape #np.swapaxes(X,1,2)
y = preprocessing.scale(np.log10(y+0.1))
idxs = np.arange(X.shape[0])
npr.seed(0)
npr.shuffle(idxs)

geneNames = np.array(genes[genes.columns[8]], dtype='S')[idxs]
X = X[idxs]
y = y[idxs]

# check that the sum is valid
test_count = 1000
valid_count = 1000
assert(test_count + valid_count <= X.shape[0])
train_count = X.shape[0] - test_count - valid_count

print('%d training sequences ' % train_count)
print('%d test sequences ' % test_count)
print('%d validation sequences ' % valid_count)
h5f = h5py.File(outfile, 'w')
i = 0
if train_count > 0:
    h5f.create_dataset('train_in'       , data=X[i:i+train_count,:], **compress_args)
    h5f.create_dataset('train_out'      , data=y[i:i+train_count], **compress_args)
    h5f.create_dataset('train_geneName' , data=geneNames[i:i+train_count], **compress_args)
i += train_count
if valid_count > 0:
    h5f.create_dataset('valid_in'       , data=X[i:i+valid_count,:], **compress_args)
    h5f.create_dataset('valid_out'      , data=y[i:i+valid_count], **compress_args)
    h5f.create_dataset('valid_geneName' , data=geneNames[i:i+valid_count], **compress_args)
i += valid_count
if test_count > 0:
    h5f.create_dataset('test_in'        , data=X[i:i+test_count,:], **compress_args)
    h5f.create_dataset('test_out'       , data=y[i:i+test_count], **compress_args)
    h5f.create_dataset('test_geneName'  , data=geneNames[i:i+test_count], **compress_args)
h5f.close()
