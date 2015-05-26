#include "dataconverter.h"

void dataHandlerFeaturesToDlib (const vec<sample_t> &h, vec<matD> &l)
{
    assert (l.size () == 0 && h.size () > 0);
    uint num_of_feat = h[0].size ();
    matD feat;
    feat.set_size (num_of_feat, 1);
    for (size_t i = 0; i < h.size (); i++)
    {
        for (size_t j = 0; j < num_of_feat; j++)
        {
            feat(j) = h[i][j];
        }
        l.push_back (feat);
    }
    assert (h.size () == l.size ());
}

void dataHandlerLabelsToDlib (const vec<sample_t> &h, vec<label_t> &l)
{
    assert (l.size () == 0 && h.size () > 0);
    for (size_t i = 0; i < h.size (); i++)
    {
        l.push_back (h[i].getLabel ());
    }
    assert (h.size () == l.size ());
}
