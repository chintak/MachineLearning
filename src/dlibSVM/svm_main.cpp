#include <iostream>
#include <string>
#include <dlib/svm.h>
#include <math.h>
#include "svmtestsuite.h"

void printHelp ();

int main(int argc, char ** argv)
{
    SVMTestSuite svm;
    std::string csv_file;
    std::string train_file;
    std::string test_file;
    double train_ratio;
    double C1 = 1.0, C2 = 1.0;
    if (argc < 5)
    {
        printHelp ();
        return 0;
    }
    if (argc >= 5)
    {
        csv_file = std::string(argv[1]);
        if (std::string(argv[2]) == "0")
        {
            train_file = std::string(argv[3]);
            test_file = std::string(argv[4]);
            std::cout << "Using training file: " << train_file
                      << " and testing file " << test_file << ".\n\n";
            svm.load (train_file, test_file);
        }
        else
        {
            train_file = std::string(argv[3]);
            train_ratio = std::stod (std::string(argv[4]));
            if (train_ratio > 1.0)
            {
                std::cout << "Using training file: " << train_file << " with "
                          << (uint) train_ratio << " training samples.\n\n";
                svm.load (train_file, (uint) train_ratio);
            }
            else if (train_ratio > 0.0)
            {
                std::cout << "Using training file: " << train_file << " with "
                          << train_ratio << " as the training ratio.\n\n";
                svm.load (train_file, train_ratio);
            }
        }
    }
    if (argc >= 6)
        C1 = std::stod (std::string(argv[5]));
    if (argc == 7)
        C2 = std::stod (std::string(argv[6]));

    std::ifstream tests (csv_file.c_str ());
    std::string line, word;

    svm.setPosC (C1);
    svm.setNegC (C2);
    while (std::getline (tests, line))
    {
        std::string predFile = line.substr (line.find_last_of (',')+1);
        if (predFile != "")
        {
            predFile = predFile.substr (predFile.find_first_not_of (' '));
            svm.predictionFile (predFile);
        }
        else
            svm.noOutput ();
        std::stringstream ss (line.substr (0, line.find_last_of (',')));
        std::vector<size_t> ff;
        std::cout << "###################################################\n"
                  <<"Testing with features: ";
        while (std::getline (ss, word, ','))
        {
            std::cout << (size_t) std::stoul (word) << ", ";
            ff.push_back ((size_t) std::stoul (word) - 1);
        }
        std::cout << "\n";
        svm.setTestMode (CUSTOM, ff);
        svm.classify ();
    }
    return 0;
}

void printHelp ()
{
    printf ("Incorrect usage: 5 required arguments and 2 optional arguments.\n");
    printf ("Usage:\n");
    std::cout << "./bin/test_svm file_specify_tests mode features_file num_of_training_samples_or_train_to_test_ratio [weight_of_class_1] [weight_of_class_-1]\n";
    printf ("\n");
    printf ("- file_specify_tests:\n"
            "\t\t\tLook at test.csv.sample. Each line represents a test case\n"
            "\t\t\twhich specifies the output file name and comma separated\n"
            "\t\t\tlist of features to be selected for training and testing.\n");
    printf ("- mode:\n"
            "\t\t\t0 to specify training_file_name and test_file_name as 3rd\n"
            "\t\t\tand 4th arguments resp.\n"
            "\t\t\t1 to specify features_file_name and\n"
            "\t\t\tnum_of_training_samples_or_train_to_test_ratio as 3rd and\n"
            "\t\t\t4th arguments resp.\n");
    printf ("- features_file:\n"
            "\t\t\tDepending on the 'mode' value this can be the training file\n"
            "\t\t\tname or the feature file name. In first case, the entire\n"
            "\t\t\tfile is used for training while in the second case, the\n"
            "\t\t\t4th argument specifies the number of samples to use as\n"
            "\t\t\ttraining samples. The rest of the samples in\n"
            "\t\t\tfeatures_file are used for testing.\n");
    printf ("- num_of_training_samples_or_train_to_test_ratio:\n"
            "\t\t\tEither the testing file or the number of samples to use\n"
            "\t\t\tout of feature_file for train/test split. This could be\n"
            "\t\t\ta value greater than 0.\n");
    printf ("- weight_of_class_1: \t [optional]\n"
            "\t\t\tValue of C for class 1 (positive class) for\n"
            "\t\t\tlinear SVM training.\n");
    printf ("- weight_of_class_-1: \t [optional]\n"
            "\t\t\tValue of C for class 2 (negative class) for\n"
            "\t\t\tlinear SVM training.\n");
    printf ("Example usage:\n");
    std::cout << "./bin/dynamicRanking/svm config/tests.csv.sample 1 config/features_05_01.txt.sample 400\n";
    std::cout << "./bin/dynamicRanking/svm config/tests.csv.sample 1 config/features_05_01.txt.sample 0.6 1.1 1\n";
    std::cout << "./bin/dynamicRanking/svm config/tests.csv.sample 0 config/features_05_01.txt.sample config/features_05_01.txt.sample\n";
}
