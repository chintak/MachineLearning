#include "svmtestsuite.h"

SVMTestSuite::SVMTestSuite() :
    nfold(3),
    C1(0),
    C2(0),
    writePred(false),
    separateTrainTestDat(false)
{}

SVMTestSuite::SVMTestSuite(const Str_t &train_file, const Str_t &test_file) :
    nfold(3),
    C1(0),
    C2(0),
    writePred(false),
    separateTrainTestDat(false)
{
    load (train_file, test_file);
}

SVMTestSuite::SVMTestSuite(const Str_t &feature_file, uint num_train_samp) :
    nfold(3),
    C1(0),
    C2(0),
    writePred(false),
    separateTrainTestDat(false)
{
    load (feature_file, num_train_samp);
}

SVMTestSuite::SVMTestSuite(const Str_t &feature_file, double train_ratio) :
    nfold(3),
    C1(0),
    C2(0),
    writePred(false),
    separateTrainTestDat(false)
{
    load (feature_file, train_ratio);
}

void SVMTestSuite::load (const Str_t &train_file, const Str_t &test_file)
{
    trainName = train_file;
    testName = test_file;
    DataHandler trainDat (train_file, 1.0);
    trainSet = trainDat.getTrainSetConst ();
    trainMean = trainDat.getTrainMeanConst ();
    trainPrec = trainDat.getTrainPrecConst ();
    numFeat = trainDat.num_feat;
    assert (trainSet.size () > 0 &&
            trainMean.size () == trainDat.num_feat &&
            trainPrec.size () == trainDat.num_feat);
    DataHandler testDat (test_file, trainMean, trainPrec);
    testSet = testDat.getTestSetConst ();
    assert (testSet.size () > 0);
    initTrainer ();
}

void SVMTestSuite::load (const Str_t &feature_file, uint num_train_samp)
{
    featureName = feature_file;
    assert (num_train_samp > 0);
    DataHandler featureDat(feature_file, num_train_samp);
    trainSet = featureDat.getTrainSetConst ();
    testSet = featureDat.getTestSetConst ();
    trainMean = featureDat.getTrainMeanConst ();
    trainPrec = featureDat.getTrainPrecConst ();
    trainRatio = featureDat.trainTestRatio;
    numFeat = featureDat.num_feat;
    assert (trainSet.size () > 0 &&
            trainMean.size () == featureDat.num_feat &&
            trainPrec.size () == featureDat.num_feat);
    initTrainer ();
}

void SVMTestSuite::load (const Str_t &feature_file, double train_ratio)
{
    assert (train_ratio > 0);
    DataHandler featureDat(feature_file, train_ratio);
    trainSet = featureDat.getTrainSetConst ();
    testSet = featureDat.getTestSetConst ();
    trainMean = featureDat.getTrainMeanConst ();
    trainPrec = featureDat.getTrainPrecConst ();
    trainRatio = featureDat.trainTestRatio;
    numFeat = featureDat.num_feat;
    assert (trainSet.size () > 0 &&
            trainMean.size () == featureDat.num_feat &&
            trainPrec.size () == featureDat.num_feat);
    initTrainer ();
}

void SVMTestSuite::initTrainer ()
{
    trainer.set_c (0);
    INIReader reader("config/ranking.ini");
    if (reader.ParseError() < 0)
    {
        std::cout << "Cannot load ranking.ini" << std::endl;
        exit (-1);
    }

    pathName = reader.Get("paths", "ClipsFolder", "");

    dlib::matrix<double> work_around;
    work_around = dlib::ones_matrix<double> (numFeat, 1);
    std::vector<dlib::matrix<double>> vw;
    vw.push_back (work_around);
    vw.push_back (work_around * -1.0);
    vw.push_back (work_around * 0.0);
    dlib::vector_normalizer<sample_type> normalizer;
    normalizer.train(vw);
    learned_function.normalizer = normalizer;
}

void SVMTestSuite::setTestMode ()
{
    setTestMode (SINGLE_USE_ALL_FEATURES, vec<size_t> ());
}

void SVMTestSuite::setTestMode (TESTMODE_t testMode,
                                const vec<size_t> &feature_set)
{
    *this << "############ New Test Mode Set ###########\n";
    vec<size_t> featureSet;
    switch (testMode)
    {
        case CUSTOM:
            featureSet = feature_set;
            *this << "- Mode set to CUSTOM\n";
            *this << "    - Using the following features for training: ";
            for (size_t i = 0; i < featureSet.size (); i++)
                *this << featureSet[i] << " ";
            *this << "\n";
            break;
        case SINGLE_USE_ALL_FEATURES:
            *this << "- Mode set to SINGLE_USE_ALL_FEATURES\n";
            for (size_t k = 0; k < numFeat; k++)
                featureSet.push_back (k);
    }

    dataHandlerToDlib (trainSet, samples, labels, featureSet);
    dataHandlerToDlib (testSet, testSamples, testLabels, featureSet);
    if (C1 == 0 || C2 == 0)
    {
        crossValidateBestC ();
        *this << "- Finding best C using cross validation.\n"
              << "Best C: " << '\t' << C1;
    }
    else
        *this << "- Using the user input \n\t- C1: " << "\t" << C1
              << "\n\t- C2: " << "\t" << C2;
    train (samples, labels);
}

void SVMTestSuite::train (const vec<sample_type> &s, const vec<label_t> &l)
{
    std::cout << "C1: " << std::setprecision (2) << std::setw (3) << C1
              << "\t\tC2: " << std::setprecision (2) << std::setw (3) << C2;
    learned_function.function = trainer.train (s, l);
}

void SVMTestSuite::setC (double C_)
{
    C1 = C_;
    C2 = C_;
    trainer.set_c_class1 (C_);
    trainer.set_c_class2 (C_);
}

void SVMTestSuite::setPosC (double C_)
{
    C1 = C_;
    trainer.set_c_class1 (C_);
}

void SVMTestSuite::setNegC (double C_)
{
    C2 = C_;
    trainer.set_c_class2 (C_);
}

void SVMTestSuite::dataHandlerToDlib (const vec<sample_t> &h,
                                      vec<sample_type> &s,
                                      vec<label_t> &l,
                                      const vec<size_t> &f)
{
    l.clear ();
    s.clear ();
    assert (h.size () > 0);
    matD feat;
    feat.set_size (f.size (), 1);
    for (size_t i = 0; i < h.size (); i++)
    {
        for (size_t j = 0; j < f.size (); j++)
        {
            feat(j) = h[i][f[j]];
        }
        s.push_back (feat);
        l.push_back (h[i].getLabel ());
    }
    assert (h.size () == s.size () && l.size () == s.size ());
}

void SVMTestSuite::crossValidateBestC ()
{
    double C_;
    float max_acc = 0.0;
    dlib::matrix<double, 1,2> acc;
    std::cout << "First performing coarse Grid Search using cross validation: " << std::endl;
    for (double C = 1; C < 10000; C *= 5)
    {
        trainer.set_c_class1 (C);
        trainer.set_c_class2 (C);

        std::cout << "C: " << std::setw(5) << C;
        acc = dlib::cross_validate_trainer(trainer, samples, labels, nfold);
        std::cout << "     cross validation accuracy: "
             << acc;
        if (acc(0) * acc(1) > max_acc)
        {
            max_acc = acc(0) + 0.5 * acc(1);
            C_ = C;
        }
    }

    std::cout << "Found C:" << std::setw(5) << C_ << "\n";
    std::cout << "Now performing fine Grid Search in the neighborhood of above C from: \n["
         << C_ - C_ / 2 << ", " << C_ + C_ / 2 << "] increment by " << C_ / 5 << std::endl;
    for (double C = C_ - C_/2; C < C_ + C_/2; C += C_ / 5)
    {
        trainer.set_c_class1 (C);
        trainer.set_c_class2 (C);
        std::cout << "C: " << C;
        acc = dlib::cross_validate_trainer(trainer, samples, labels, nfold);
        std::cout << "     cross validation accuracy: "
             << acc;
        if (acc(0) * acc(1) > max_acc)
        {
            max_acc = acc(0) + 0.5 * acc(1);
            C_ = C;
        }
    }
    std::cout << "Best C:" << std::setw(5) << C_ << "\n";
    trainer.set_c (C_);
    C1 = C_;
    C2 = C_;
}

void SVMTestSuite::classify ()
{
    classify (testSamples, testLabels);
}

void SVMTestSuite::classify (const vec<sample_type> &s, const vec<label_t> &l)
{
    Str_t ing = "interesting";
    Str_t ning = "not_interesting";
    *this << "\n#####################" << "\nStarting classification:"
          << "\n#####################";
    *this << "\n---------------------------------------------------------------------------------------------------------------";
    *this << "\nSr #\t\t|\t\t" << "Prediction" << "\t|\t" << "Original"
          << "\t|\t\t" << "Comments";
    *this << "\n---------------------------------------------------------------------------------------------------------------";
    assert ((s.size () > 0 && l.size () > 0) ||
            !(std::cout << "Test set size 0. Run setTestMode first.\n"));
    float epos = 0, eneg = 0, tpos = 0, tneg = 0;
    label_t p = 0;
    for (size_t k = 0; k < s.size (); k++)
    {
        p = learned_function(s[k]);
        *this << "\n#" << k+1 << "\t\t|\t\t" << ((int) (p * 10000)) / 100000.0 + 0.000011
              << "\t\t:\t\t" << l[k] << "\t\t|\t\t" << testSet[k].getComments ();
        if (l[k] < 0)
        {
            tneg += 1;
            if (p > 0)
            {
                eneg += 1;
                *this << "\t" << "FP";
            }
        }
        else if (l[k] > 0)
        {
            tpos += 1;
            if (p < 0)
            {
                epos += 1;
                *this << "\t" << "FN";
            }
        }
        else if (l[k] == 0)
        {
            std::string mp4FileName = testSet[k].getComments ();
            std::string jpgFileName = mp4FileName.substr(0, mp4FileName.find_first_of ('.')) + ".jpg";
            if (p > 0)
            {
                tpos += 1;
                moveFile (mp4FileName, pathName, "", ing);
                moveFile (jpgFileName, pathName, "", ing);
            }
            else
            {
                tneg += 1;
                moveFile (mp4FileName, pathName, "", ning);
                moveFile (jpgFileName, pathName, "", ning);
            }
        }
    }
    *this << "\n% of correctly classified +1 class: " << 1.0 - epos / tpos
          << "\n% of correctly classified -1 class: " << 1.0 - eneg / tneg;

    *this << "\nIncorrect +1 classified: " << epos << " / " << tpos
          << "\nIncorrect -1 classified: " << eneg << " / " << tneg;
//    printf ("Incorrect +ve : %f / %f\nIncorrect -ve : %f / %f\n", epos, tpos, eneg, tneg);
    std::cout << "\nFP/P : " << std::setprecision (3)
              << std::setw(5) << epos / tpos << "\n";
    std::cout << "FN/N : " << std::setprecision (3)
              << std::setw(5) << eneg / tneg << "\n";
    printf ("Done.\n");
}

void moveFile (const Str_t &f, const Str_t p, const Str_t &s, const Str_t &d)
{
    std::stringstream com;
    com << p << '/' << s << '/' << f;
    if (!fileExists (com.str ()))
        return;
    com.str(std::string());
    com << "mkdir -p " << p << '/' << d << '/';
    if (system (com.str ().c_str ()));

    com.str(std::string());
    com << "mv " << p << '/' << s << '/' << f << ' '
        << p << '/' << d << '/' << f << "";
    if (system (com.str ().c_str ()));
}

void SVMTestSuite::predictionFile (const Str_t &outname)
{
    if (writePred)
        logP.close ();
    writePred = true;
    logP.open (outname.c_str ());
    if (!logP.is_open ())
    {
        std::cout << "## Error opening file " << outname << "\n"
                  << "## Aborting prediction output.\n";
        writePred = false;
    }
    else
    {
        logP << "**************\n";
        if (separateTrainTestDat)
        {
            logP << "- Training file name: " << trainName << "\n";
            logP << "- Testing file name: " << testName << "\n";
        }
        else
        {
            logP << "- All samples file name: " << featureName << "\n";
            logP << "- Train to test ratio: " << trainRatio << "\n";
        }
    }
}

SVMTestSuite& SVMTestSuite::operator<< (const double &s)
{
    if (writePred)
        logP << s << ' ';
    return *this;
}

SVMTestSuite& SVMTestSuite::operator<< (const int &s)
{
    if (writePred)
        logP << s << ' ';
    return *this;
}

SVMTestSuite& SVMTestSuite::operator<< (const size_t &s)
{
    if (writePred)
        logP << s << ' ';
    return *this;
}

SVMTestSuite& SVMTestSuite::operator<< (const std::string &s)
{
    if (writePred)
        logP << s << ' ';
    return *this;
}
