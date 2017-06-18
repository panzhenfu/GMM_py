
1.安装依赖库
sudo apt-get install libopenblas-dev
sudo apt-get install liblapack-dev
sudo apt-get install libarpack2-dev
sudo apt-get install libsuperlu-dev

2.下载安装armadillo线性代数库　(1)cmake (2)make (3) sudo make install

3.编译：g++ GMM_cplus.cpp -o GMM.bin -O2 -larmadillo