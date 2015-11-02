# bash 
cd data
wget http://www.cs.toronto.edu/~nitish/multimodal/flickr_data.tar.gz
tar -zxvf flickr_data.tar.gz
rm flickr_data.tar.gz

cd flickr
# convert .npy files into .mat files
python data_npy2mat.py

cd ..
cd ..
