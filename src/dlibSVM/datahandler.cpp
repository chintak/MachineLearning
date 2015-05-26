#include "datahandler.h"


DataHandler::DataHandler (const Str_t &filename) :
    DataHandler (filename, 1.0)
{}

DataHandler::DataHandler (const Str_t &filename, uint train_num_samples) :
    num_feat(0),
    samples(vecS_t ()),
    num_pos(0),
    num_neg(0),
    num_train(0)
{
    if (train_num_samples == 0) DataHandler(filename, 0.0);
    getData (filename);
    trainTestSplit (train_num_samples);
}

DataHandler::DataHandler (const Str_t &filename, double train_to_test_ratio) :
    num_feat(0),
    samples(vecS_t ()),
    num_pos(0),
    num_neg(0),
    num_train(0)
{
    getData (filename);
    trainTestSplit (train_to_test_ratio);
}

DataHandler::DataHandler (const Str_t &filename, const vecF_t &mu, const vecF_t &prec) :
    num_feat(0),
    samples(vecS_t ()),
    num_pos(0),
    num_neg(0),
    num_train(0)
{
    assert (mu.size () > 0 && prec.size () > 0);
    trainMean = mu;
    trainPrec = prec;
    getData (filename);
    trainTestSplit (0.0);
}

DataHandler::~DataHandler()
{}

/* **************************************************************************
 * Private member functions
 * **************************************************************************
*/
void DataHandler::getData (const Str_t &filename)
{
    printf ("********** DataHandler processing **********\n");
    getNumFeatures (filename);
    fileReader (filename);
}

void DataHandler::getNumFeatures (const Str_t &filename)
{
    std::ifstream f(filename);
    if(!f.is_open ())
    {
        std::cout << "Error reading file: " << filename << "\n";
        return;
    }
    Str_t line;
    while (std::getline(f, line, '\n') && num_feat == 0)
    {
        if (line[0] == '#') continue;
        num_feat = std::stoul (
                line.substr (
                    line.substr (
                        0, line.substr (
                            0, line.find_first_of ('#')).find_last_of (':')).find_last_of (' ') + 1));
    // TODO: find out number of samples to preallocate contiguous memory
    }
    f.close ();
    assert (num_feat > 0);
}

/// Determine the splits and assert that the split hasn't already been made
void DataHandler::trainTestSplit (uint train_num_samples)
{
    uint min_samp = std::min(num_neg, num_pos);
    assert (samples.size () > 0 && trainSet.size () == 0 && testSet.size () == 0);  // TODO: remove this
    assert (train_num_samples <= min_samp && train_num_samples >= 0);
    trainTestRatio = train_num_samples / (double) min_samp;
    num_train = (uint) train_num_samples;
    populateNormalizeTrainTest ();
}

void DataHandler::trainTestSplit (double train_to_test_ratio)
{
    assert (samples.size () > 0 && trainSet.size () == 0 && testSet.size () == 0);
    assert (train_to_test_ratio <= 1.0 && train_to_test_ratio >= 0.0);
    trainTestRatio = train_to_test_ratio;
    num_train = (num_pos < num_neg) ? (uint)(num_pos * train_to_test_ratio + 0.5) :
                                      (uint)(num_neg * train_to_test_ratio + 0.5);
    populateNormalizeTrainTest ();
}

void DataHandler::populateNormalizeTrainTest ()
{
    assert (trainTestRatio > 0 ||
            (trainTestRatio == 0 && trainMean.size () > 0 && trainPrec.size () > 0));
    if (trainTestRatio < 1.0 && trainTestRatio > 0.0)
        randomizeSamples (samples);
    else if (trainTestRatio == 1)
        printf ("Training Mode selected.\n");
    else
        printf ("Testing Mode selected.\n");
    populateTrainTest ();
    if (trainMean.size () == 0 && trainPrec.size () == 0)
        trainSetNormStats ();
    normalizeSet (trainSet, trainMean, trainPrec);
    normalizeSet (testSet, trainMean, trainPrec);
    printf ("********** Finished processing **********\n");
}

/// Populate training set and testing set
void DataHandler::populateTrainTest ()
{
    size_t trainingSetSize = 2 * num_train;
    size_t trainPos = 0, trainNeg = 0;
    label_t lab = 0;
    vecS_t::iterator itS = samples.begin ();
    sample_t sample;
    sample.reserve (num_feat);
    while (itS != samples.end ())
    {
        lab = itS->getLabel ();
        if (trainSet.size () < trainingSetSize && lab > 0 && trainPos < num_train)
        {
            trainSet.push_back (*itS);
            trainPos++;
        }
        else if (trainSet.size () < trainingSetSize && lab <= 0 && trainNeg < num_train)
        {
            trainSet.push_back (*itS);
            trainNeg++;
        }
        else
        {
            testSet.push_back (*itS);
        }
        itS = samples.erase (itS);
    }
    if (trainTestRatio == 0.0)
        printf ("- Number of samples in testing set: %lu\n", testSet.size ());
    else if (trainTestRatio == 1.0)
        printf ("- Number of samples in training set: %lu\n", trainSet.size ());
    else
    {
        printf ("- Number of samples in training set: %lu\n", trainSet.size ());
        printf ("- Number of samples in testing set: %lu\n", testSet.size ());
    }
    assert (samples.size () == 0);  // Free memory
    assert ((trainSet.size () > 0 && trainTestRatio > 0) ||
            (trainSet.size () == 0 && trainTestRatio == 0));
}

void DataHandler::_sum (const vecF_t &x, vecF_t &y)
{
    assert (x.size () == y.size () ||
            !(std::cerr << "x size: " << x.size() << " and y size: " << y.size() << "\n"));
    for (uint i = 0; i < x.size (); i++)
        y[i] += x[i];
}

void DataHandler::_accumulate (const vecS_t &x, vecF_t &acc)
{
    vecS_t::const_iterator itS = x.begin ();
    while (itS != x.end ())
    {
        _sum(itS->getFeatures (), acc);
        ++itS;
    }
}

void DataHandler::mean (const vecS_t &x, vecF_t &mu)
{
    assert (mu.size () == num_feat);
    for (uint i = 0; i < mu.size (); i++)
        mu[i] = 0.0;
    _accumulate(x, mu);
    for (uint i = 0; i < mu.size (); i++)
        mu[i] = mu[i] / (x.size() - 1.0);
}

void DataHandler::precision (const vecS_t &x, vecF_t &prec, const vecF_t &mean)
{
    assert (prec.size () == num_feat);
    for (uint i = 0; i < prec.size (); i++)
        prec[i] = 0.0;
    for (uint i = 0; i < x.size (); i++)
        for (uint j = 0; j < mean.size (); j++)
            prec[j] += (x[i][j] - mean[j]) * (x[i][j] - mean[j]);
    for (uint i = 0; i < prec.size (); i++)
        prec[i] = 1. / std::sqrt (prec[i] / (x.size() - 1.0));
}

void DataHandler::trainSetNormStats ()
{
    assert (trainMean.size () == 0 && trainPrec.size () == 0);
    trainMean.resize (num_feat);
    trainPrec.resize (num_feat);
    mean (trainSet, trainMean);
    precision (trainSet, trainPrec, trainMean);
    printf ("- Computed training data statistics.\n");
}

void DataHandler::normalizeSet (vecS_t &x, vecF_t &mu, vecF_t &prec)
{
    if (x.size () == 0) return;
    for (uint i = 0; i < x.size (); i++)
        for (uint j = 0; j < mu.size (); j++)
            x[i][j] = (x[i][j] - mu[j]) * prec[j];
    printf ("- Data normalized.\n");
}

void DataHandler::randomizeSamples (vecS_t &x)
{
    std::srand (12345);
    size_t n = x.size () - 1;
    size_t idx = 0;
    while (n > 0)
    {
        idx = rand () % (n + 1);
        std::swap (x[idx], x[n]);
        n--;
    }
    printf ("- Data randomized.\n");
}

unsigned int DataHandler::readLineSVMLightFormat (const Str_t &txt, sample_t &feat)
{
    // No support for sparse input. TODO: enable this
//    long int ind = 0;
    std::stringstream ss(txt);
    Str_t comment;
    Str_t word;
    std::size_t idx;

    std::getline(ss, word, ' ');
    label_t lab = std::stod (word, NULL);
    if (lab > 0)
        num_pos++;
    else
        num_neg++;
    feat.push_lab (lab);
    while(std::getline(ss, word, ' '))
    {
        if (word[0] == '#')
        {
            std::getline(ss, comment);
            break;
        }
        idx = word.find (':');
        if (idx <= 0) continue;

//        ind = std::stol (word.substr(0, idx));
        feat.push_back (std::stod (word.substr(idx+1), NULL));
    }
    feat.push_comment (comment);
    return 0;
}

void DataHandler::fileReader (Str_t filename)
{
    std::ifstream f(filename);
    if(!f.is_open ())
    {
        std::cout << "Error reading file: " << filename << "\n";
        return;
    }
    Str_t line;

    sample_t feat;
    feat.reserve (num_feat);

    while (std::getline(f, line, '\n'))
    {
        if (line[0] == '#') continue;
        feat.clear ();
        readLineSVMLightFormat (line, feat);
        samples.push_back (feat);
    }
    printf("Finished reading %s file.\n", filename.c_str ());
    std::cout << "Total number of examples read: " << samples.size () << "\n";
    f.close ();
}

//Utility functions
void printSet (const vecS_t &x)
{
    std::cout << std::fixed << std::setprecision (6);
    for (size_t i = 0; i < x.size (); i++)
    {
        std::cout << "#" << std::setw(7) << i+1 << " | ";
        for (size_t j = 0; j < x[i].size (); j++)
            std::cout << std::setw(10) << (double) x[i][j] << " ";
        std::cout << "  |  " << std::setw(3) << (int) x[i].getLabel () << " ";
        std::cout << "  |  " << x[i].getComments () << " ";
        printf ("\n");
    }
}
