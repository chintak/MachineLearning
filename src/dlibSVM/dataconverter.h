#ifndef DATACONVERTER_H
#define DATACONVERTER_H

#include "datahandler.h"
#include <dlib/matrix.h>

/* This file contains some utility functions to convert 'vector<vector<double>>'
 * format used for storing data sets in DataHandler class to format accepted by
 * different machine libraries for their algorithms.
*/
template<typename T>
using vec = std::vector<T>;

template<typename T>
using mat = dlib::matrix<T>;

typedef vec<double> vecD;
typedef mat<double> matD;
typedef unsigned int uint;

void dataHandlerFeaturesToDlib (const vec<sample_t> &h, vec<matD> &l);
void dataHandlerLabelsToDlib (const vec<sample_t> &h, vec<label_t> &l);

#endif // DATACONVERTER_H
