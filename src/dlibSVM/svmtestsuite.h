#ifndef SVMTESTSUITE_H
#define SVMTESTSUITE_H

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <dlib/svm/cross_validate_assignment_trainer.h>
#include <dlib/svm/svm_c_linear_trainer.h>
#include <sys/stat.h>
#include "datahandler.h"
#include "INIReader.h"


template<typename T>
using vec = std::vector<T>;
template<typename T>
using mat = dlib::matrix<T>;
typedef unsigned int uint;

typedef vec<double> vecD_t;
typedef std::string Str_t;
typedef double label_t;

typedef dlib::matrix<double> matD;
typedef dlib::matrix<double> sample_type;
typedef dlib::linear_kernel<sample_type> kernel_type;
typedef dlib::decision_function<kernel_type> dec_funct_type;
typedef dlib::normalized_function<dec_funct_type> funct_type;

typedef enum testMode {
    SINGLE_USE_ALL_FEATURES,
    CUSTOM
} TESTMODE_t;

class SVMTestSuite
{
public:
    SVMTestSuite();
    SVMTestSuite(const Str_t &train_file, const Str_t &test_file);
    SVMTestSuite(const Str_t &feature_file, double train_ratio);
    SVMTestSuite(const Str_t &feature_file, uint num_train_samp);
    void load (const Str_t &train_file, const Str_t &test_file);
    void load (const Str_t &feature_file, uint num_train_samp);
    void load (const Str_t &feature_file, double train_ratio);
    ~SVMTestSuite()
    {
        if (writePred)
        {
            logP << "\n**************\n";
            logP.close ();
        }
    }
    void setTestMode ();
    void setTestMode (TESTMODE_t mode, const vec<size_t> &feature_set);
    void predictionFile (const Str_t &outname);
    void noOutput () { writePred = false; }
    void classify ();
    void classify (const vec<sample_type> &s, const vec<label_t> &l);
    SVMTestSuite& operator<< (const std::string &s);
    SVMTestSuite& operator<< (const double &s);
    SVMTestSuite& operator<< (const int &s);
    SVMTestSuite& operator<< (const size_t &s);
    void setC (double C_);
    void setNegC (double C_);
    void setPosC (double C_);

    std::ofstream logP;

private:
    void train (const vec<sample_type> &s, const vec<label_t> &l);
    void dataHandlerToDlib (const vec<sample_t> &h, vec<sample_type> &s,
                            vec<label_t> &l, const vec<size_t> &f);
    void crossValidateBestC ();
    void initTrainer ();

    vecS_t trainSet;
    vecS_t testSet;
    vecF_t trainMean;
    vecF_t trainPrec;
    vec<sample_type> samples, testSamples;
    vec<label_t> labels, testLabels;
    funct_type learned_function;
    dlib::svm_c_linear_trainer<kernel_type> trainer;
    uint nfold;
    uint numFeat;
    Str_t trainName;
    Str_t testName;
    Str_t featureName;
    double trainRatio;
    double C1;
    double C2;
    bool writePred;
    bool separateTrainTestDat;
    std::string pathName;
};

void moveFile (const Str_t &f, const Str_t p, const Str_t &s, const Str_t &d);
inline bool fileExists (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

#endif // SVMTESTSUITE_H
