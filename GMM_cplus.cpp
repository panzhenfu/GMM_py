//
// Created by panzhenfu on 17-6-17.
//
#include <iostream>
#include <armadillo>
#include <math.h>
#include <assert.h>
#include <fstream>
float Gaussion_DN(arma::fmat X, arma::fmat U_mean, arma::fmat Cov){
    int Dim = X.n_rows;
    arma::fmat Y = X - U_mean;
    arma::fmat temp = Y.t() * (Cov + arma::eye<arma::fmat>(Dim,Dim)*0.01).i() * Y;
    //temp.print("temp:");
    float temp1 = (1.0/std::pow(2*M_PI,Dim/2))*(1.0/std::pow(arma::det((Cov + arma::eye<arma::fmat>(Dim,Dim)*0.01)),0.5));
    float  tt= temp(0,0);
    float result = temp1*std::exp(-0.5 * temp(0,0));
    return result;
}

//计算均值
arma::fmat CalMean(arma::fmat X){
    int Dim = X.n_rows;
    int Num = X.n_cols;
    std::cout << "Dim :" << Dim << " Num:" << Num <<std::endl;
    arma::fmat Mean(Dim,1);
    Mean = arma::sum(X, 1);
    Mean /= Num;
    return Mean;
}

//计算似然函数值
float maxLikelyhoood(arma::fmat Xn,arma::fmat Pik,arma::fmat Uk,arma::field<arma::fmat> Cov, arma::fmat &probility_mat){
    int Dim = Xn.n_rows;
    int Num = Xn.n_cols;
    int Dim_k = Uk.n_rows;
    int K = Uk.n_cols;
    assert(Dim == Dim_k);
    probility_mat.set_size(Num,K);
    double likelyhood = 0.0;
    for(int n_iter = 0;n_iter < Num; ++n_iter){
        double temp = 0.0;
        for(int k_iter = 0; k_iter < K;++k_iter){
            float gaussian = Gaussion_DN(Xn.col(n_iter),Uk.col(k_iter),Cov(k_iter));
            //std::cout << "gaussian:"<< gaussian<<std::endl;
            probility_mat(n_iter,k_iter) = gaussian;
            temp += Pik(0,k_iter) * gaussian;
        }
        likelyhood += std::log(temp);
    }
    return float(likelyhood);
}

//用ＥＭ算法训练混合高斯模型
bool EMforMixGaussian(arma::fmat InputDate,int K, int MaxIter,
                      arma::fmat &Pi_cof,arma::fmat &Uk, arma::field<arma::fmat> &cov_list){

    int Dim = InputDate.n_rows;
    int Num = InputDate.n_cols;
   //初始化Pik
    Pi_cof.set_size(1,K);
    Pi_cof.fill(1.0);
    Pi_cof *=  (1.0/float(K));
    Pi_cof.print("Pi_cof:");
    //初始化Uk
    arma::fmat X_mean = CalMean(InputDate);
    Uk.set_size(Dim,K);
    for(int k_iter= 0;k_iter<K;k_iter++){
       // Uk.col(k_iter) = X_mean + arma::randu<arma::fmat>(Dim,1);
        Uk.col(k_iter) = arma::randu<arma::fmat>(Dim,1);
    }
    Uk.print("Uk:");

    //初始化Ｋ个协方差矩阵列表
    cov_list.set_size(K);
    for(int k_iter=0; k_iter<K; k_iter++){
        cov_list(k_iter) = arma::eye<arma::fmat>(Dim,Dim)*0.0001;
    }
    cov_list.print("cov_list:");

    arma::fmat probility;//此处没用到
    float likelyhood_old = maxLikelyhoood(InputDate,Pi_cof,Uk,cov_list,probility);
    std::cout << "likelyhood_old: " << likelyhood_old<<std::endl;
    float likelyhood_new = 0.0;
    int currentIter = 0;
    while(true){
        currentIter++;
        arma::fmat rZnk = arma::zeros<arma::fmat>(Num,K);
        arma::fmat denominator = arma::zeros<arma::fmat>(Num,1);
        // rZnk=pi_k*Gaussian(Xn|uk,Cov_k)/sum(pi_j*Gaussian(Xn|uj,Cov_j))
        //待改善
        for(int n_iter=0; n_iter < Num; n_iter++){
            for(int k_iter=0;k_iter<K;k_iter++){
                rZnk(n_iter,k_iter) = Pi_cof(0,k_iter) * Gaussion_DN(InputDate.col(n_iter),Uk.col(k_iter),cov_list(k_iter));
                denominator(n_iter,0) += rZnk(n_iter,k_iter);
            }
            for(int k_iter=0;k_iter<K;k_iter++){
                rZnk(n_iter,k_iter) /= denominator(n_iter,0);
            }
        }

        //Nk=sum(rZnk)　新的ＰＩ
        arma::fmat Nk = arma::zeros<arma::fmat>(1,K);
        arma::fmat pi_new = arma::zeros<arma::fmat>(1,K);
        Nk = arma::sum(rZnk);
        pi_new = Nk / float(Num);
        //等效代码
//        for(int k_iter=0; k_iter<K;k_iter++){
//            for(int n_iter=0;n_iter < Num;n_iter++){
//                Nk(0,k_iter) += rZnk(n_iter,k_iter);
//            }
//            pi_new(0,k_iter) = Nk(0,k_iter)/float(Num);
//        }

        //Uk_new=(1/sum(rZnk))*sum(rZnk*Xn)　新的均值
        arma::fmat Uk_new = arma::zeros<arma::fmat>(Dim,K);
        Uk_new = InputDate * rZnk;
        for(int d_iter = 0;d_iter < Dim; d_iter++){
            Uk_new.row(d_iter) /= Nk;
        }
//等效代码
//        for(int k_iter=0;k_iter<K;k_iter++){
//            for(int n_iter=0;n_iter<Num;n_iter++){
//                Uk_new.col(k_iter) += (1.0/float(Nk(0,k_iter)))*rZnk(n_iter,k_iter)*InputDate.col(n_iter);
//            }
//        }

        //cov_list_new 新的协方差
        arma::field<arma::fmat> cov_list_new(K);
        for(int k_iter=0; k_iter < K; k_iter++){
            arma::fmat X_cov_new = arma::zeros<arma::fmat>(Dim,Dim);
            for(int n_iter =0;n_iter<Num;n_iter++){
                arma::fmat temp = InputDate.col(n_iter) - Uk_new.col(k_iter);
                X_cov_new += (1.0/float(Nk(0,k_iter)))*rZnk(n_iter,k_iter)* temp*temp.t();
            }
            cov_list_new(k_iter) = X_cov_new;
        }

        likelyhood_new = maxLikelyhoood(InputDate,pi_new,Uk_new,cov_list_new,probility);
        std::cout << "likelyhood_new : " << likelyhood_new <<  " " << currentIter<< std::endl;

        //break the recurrent
        if(likelyhood_old >= likelyhood_new || currentIter > MaxIter){
            break;
        }
        Uk =Uk_new;
        Pi_cof = pi_new;
        cov_list = cov_list_new;
        likelyhood_old = likelyhood_new;
    }


    return true;
}

int main(int argc, char** argv)
{
    /*
    //Gaussion_DN test
    std::vector<float> X_;
    X_.push_back(0.5);
    X_.push_back(3.0);
    X_.push_back(2.1);
    arma::fmat X(X_);
    X.print("X:");

    arma::fmat U(3,1);
    U << 1.0 <<arma::endr
             <<1.0<<arma::endr
                  <<1.0<<arma::endr;
    U.print("U:");
    arma::fmat cov(3,3);
    cov << 1.0 << 0.0 << 0.0 << arma::endr
        << 0.0 << 1.0 << 0.0 << arma::endr
        << 0.0 << 0.0 << 1.0 << arma::endr;
    cov.print("cov:");
    float result = Gaussion_DN(X, U,cov);
    std::cout << "result: "<< result <<std::endl;

    //CalMean test
    float X1[] = {0.5,3.0,2.1,0.4,2.0,2.0,0.3,1.0,1.9};
    arma::fmat X11(&X1[0],3,3);
    X11.print("X11:");
    U = CalMean(X11);
    U.print("U:");

    //maxLikelyhoood test
    float X2[] = {0.5,3.0,2.1,0.4,2.0,2.0,0.3,1.0,1.9,1.0,3.2,1.8,2.0,1.6,3.5};
    arma::fmat X22(&X2[0],3,5);
    X22.print("X11:");
    arma::fmat Pi_cof{0.5,0.2,0.3};
    Pi_cof.print("Pi_cof:");
    arma::fmat Uk{{0.5,0.3,3.0},{3.0,2.1,2.1},{2.1,3.0,1.4}};
    Uk.print("Uk:");
    arma::field<arma::fmat> cov_list(3);
    arma::fmat cov1{{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}};
    arma::fmat cov2{{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}};
    arma::fmat cov3{{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}};
    cov_list(0) = cov1;
    cov_list(1) = cov2;
    cov_list(2) = cov3;
    arma::fmat pro;
    float likelyhood = maxLikelyhoood(X22,Pi_cof,Uk,cov_list,pro);
    std::cout << "likelyhood:" << likelyhood << std::endl;
    pro.print("pro:");

    //EMforMixGaussian test
    arma::mat A = arma::randu<arma::mat>(2,3);
    arma::mat B = A;
    B(0,0) = 100;
    A.print("A:");
    A *= 10;
    A.print("A:");
    B.print("B:");
    arma::field<arma::fmat> C = cov_list;
    cov_list(1).at(0.0) = 100;
    cov_list.print("cov_list:");
    C.print("C:");
*/
    std::vector<std::vector<float >> data;
    std::ifstream infile("filename.txt");
    if(infile){
        std::string line;
        while(std::getline(infile,line)){
            int ss_index = line.find(' ');
            float a = atof(line.substr(0,ss_index).c_str());
            float b = atof(line.substr(ss_index+1,line.size()-1).c_str());
            std::cout << "a :"<<a <<"  b: "<<b << std::endl;
            std::vector<float > sub_data;
            sub_data.push_back(a);
            sub_data.push_back(b);
            data.push_back(sub_data);
        }
    }
    infile.close();

    float *InputData = (float *)malloc(data.size()*2*sizeof(float));
    memset(InputData,0,data.size()*2*sizeof(float));
    for(int i=0; i< data.size(); i++){
        InputData[i*2] = data[i].at(0);InputData[i*2 +1 ] = data[i].at(1);
    }

    arma::fmat Input_mat(InputData,2,data.size());
    Input_mat.print("Input_mat:");

    //归一化
//    arma::fmat min = arma::min(Input_mat,1);
//    arma::fmat max = arma::max(Input_mat,1);
//    for (int i = 0; i <Input_mat.n_cols;i++ ){
//        Input_mat.col(i) = (Input_mat.col(i) - min)/(max-min);
//    }
//    Input_mat.print("eeeee:");


    int Num = data.size();
    int K = 4;
    int iter_num = 5000;
    arma::fmat Pi, Uk1;
    arma::field<arma::fmat> list_cov;

    bool flag = EMforMixGaussian(Input_mat,K,iter_num,Pi,Uk1,list_cov);

    Pi.print("Pi:");
    Uk1.print("Uk1:");

    list_cov.print("list_cov");

    //------
    free(InputData);

    return 0;
}