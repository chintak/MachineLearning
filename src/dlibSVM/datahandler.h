#ifndef DATAHANDLER_H
#define DATAHANDLER_H

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


// Define shorthands for commonly used types
template<typename T>
using vec = std::vector<T>;

typedef unsigned int uint;
typedef double feature_t;
typedef vec<feature_t> vecF_t;
typedef std::string Str_t;
typedef double label_t;

typedef struct sample
{
public:
    sample () :
        feats (vecF_t ()),
        lab (0),
        comment ("")
    {}
    ~sample () {}
    inline const vecF_t & getFeatures () { return feats; }
    inline const label_t & getLabel () { return lab; }
    inline const Str_t & getComments () { return comment; }
    // Const members
    inline const vecF_t & getFeatures () const { return feats; }
    inline const label_t & getLabel () const { return lab; }
    inline const Str_t & getComments () const { return comment; }
    // Wrapper to the feature vector
    inline size_t numFeatures () { return feats.size (); }
    inline size_t size () { return feats.size (); }
    inline const size_t size () const { return feats.size (); }
    inline void clear () { feats.clear (); }
    inline void reserve (size_t t) { feats.reserve (t); }
    inline void resize (size_t t) { feats.resize (t); }
    // Add data
    inline void push_back (feature_t f) { feats.push_back (f); }
    inline void push_lab (label_t l) { lab = l; }
    inline void push_comment (Str_t c) { comment = c; }
    // Wrapper to access particular values of feats vector
    inline feature_t & operator[] (size_t t) { return feats[t]; }
    inline const feature_t & operator[] (size_t t) const { return feats[t]; }

    vecF_t feats;
    label_t lab;
    Str_t comment;
} sample_t;

typedef vec<sample_t> vecS_t;

/* Class for handling the dataset requirements
 *
 * It has the following abilities:
 * - Read all samples and labels from a single file
 * - Randomize the dataset
 * - Split into training (trainSet, trainLab) and testing (testSet, testLab) dataset
 * - Calculate the mean and precision for all features in the training dataset
 * - Standard normalize the training set
 * - Standard normalize the testing set
 *
 * Parameters:
 * - filename :             name of the file to be read
 * - train_to_test_ratio :  If the samples have unbalanced class distribution,
 *                          then it finds out the number of samples/class wrt
 *                          to the min samples/class.
 *          Ensures : training set has balanced class prob
 * - mu :                   Vector of doubles with same size as number of features.
 * - prec :                 Vector of doubles with same size as number of features
 *                          If 'mu' and 'prec' are specified, train_to_test_ratio
 *                          is assumed to be 0.0, i.e. the filename specifies
 *                          the samples for testing data.
 *                          After reading the data, only normalization is performed
 *                          using 'mu' as mean and 'prec' as precision,
 *                          randomization is skipped.
 *
 * Usage:
 * - Specify train_to_test_ratio = 1.0, if you want to read the data as
 *   training data. In this case, randomization is skipped. The mean and prec
 *   of the training data set is computed.
 * - Specify train_to_test_ratio = 0.0 and also 'mu' and 'prec', if you want to
 *   read the data as testing data. In this case, randomization is skipped and the
 *   input mean and prec is used for test data set normalization.
*/
class DataHandler
{
public:
    DataHandler () {}
    DataHandler (const Str_t &filename);
    DataHandler (const Str_t &filename, double train_to_test_ratio);
    DataHandler (const Str_t &filename, uint train_num_samples);
    DataHandler (const Str_t &filename, const vecF_t &mu, const vecF_t &prec);
    ~DataHandler ();
    const vecS_t & getTrainSetConst ()
    { assert (trainTestRatio > 0); return trainSet; }

    const vecS_t & getTestSetConst ()
    { assert (trainTestRatio < 1); return testSet; }

    const vecF_t & getTrainMeanConst ()
    { return trainMean; }

    const vecF_t & getTrainPrecConst ()
    { return trainPrec; }
    // Utility functions
    void printSet (const vecS_t &x);
    void printSet (const vecF_t &x);

    uint num_feat;
    double trainTestRatio;

private:
    void getData (const Str_t &filename);
    unsigned int readLineSVMLightFormat (const Str_t &txt, sample_t &feat);
    void fileReader (Str_t filename);
    void getNumFeatures (const Str_t &filename);
    void randomizeSamples (vecS_t &x);
    void populateTrainTest ();
    void trainTestSplit (uint train_num_samples);
    void trainTestSplit (double train_to_test_ratio);
    void trainSetNormStats ();
    void populateNormalizeTrainTest ();
    void normalizeSet (vecS_t &x, vecF_t &mu, vecF_t &prec);

    void _sum (const vecF_t &x, vecF_t &y);
    void _accumulate (const vecS_t &x, vecF_t &acc);
    void mean (const vecS_t &x, vecF_t &mu);
    void precision (const vecS_t &x, vecF_t &prec, const vecF_t &mean);
    // Variables
    vecS_t samples;
    vecS_t trainSet;
    vecS_t testSet;
    uint num_pos;
    uint num_neg;
    uint num_train;
    vecF_t trainMean;
    vecF_t trainPrec;
};

// Utility function
void printSet (const vecS_t &x);

template <typename T>
void printVect (const std::vector<T> &x)
{
    std::cout << std::fixed << std::setprecision (6);
    for (size_t i = 0; i < x.size (); i++)
        std::cout << std::setw(12) << (double) x[i] << " ";
    printf ("\n");
}

template <typename T>
std::string & stringify (T c, std::string &s)
{
    std::stringstream ss;
    ss << (T) c;
    s = ss.str ();
    return s;
}

#endif // DATAHANDLER_H
