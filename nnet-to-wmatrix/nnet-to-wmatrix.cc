#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <vector>
#include "file-utils.h"

#define SEPS " \t\n"

class Matrix {
public:
    float *data_;
    int nr_;
    int nc_;

    Matrix() : 
        data_(NULL), 
        nr_(0), 
        nc_(0) 
    { }
    ~Matrix() {
        free(data_);
        data_ = NULL;
        nr_ = 0;
        nc_ = 0;
    }
    void Init(int nr, int nc) {
        assert(data_ == NULL);
        data_ = (float *) malloc(nr * nc * sizeof(float));
        assert(data_ != NULL);
        nr_ = nr;
        nc_ = nc;
    }
};

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: nnet-to-wmatrix (i)kaldi.nnet.txt (o)WMatrix\n");
        exit(0);
    }
    const char * imodel_filename = argv[1];
    const char * omodel_filename = argv[2];
    FILE *ifp = fopen(imodel_filename, "r");  assert(ifp != NULL);
    FILE *ofp = fopen(omodel_filename, "w+"); assert(ofp != NULL);

    char tok[256];
    int tok_len = 0;

    int idim, odim;
    // useful dims
    int fbank_dim = 40;
    int delta = 3;        // fbank + delta + delta-delta
    int nsplice = 11;

    int nband = 8;        // subbands
    int band_shift = 5;   // nband * band_shift = fbankdim (e.g 8 * 5 = 40)
                          // exception: the last subband has a band shift of 4
    int band_width = 6;   // band_width(6) > band_shift(5), one bank overlapping

    int patch_dim = 4;    // filter size along frequency axis
    int patch_step = 1;   // filter shift along frequency axis
    int patch_stride = band_width;
   
    int filter_size = patch_dim * delta * nsplice;

    int band_points = band_width * delta * nsplice;  // number of points in a band
    int pool_size = 1 + (band_width - patch_dim) / patch_step;

    expect_token(ifp, SEPS, "<Nnet>");

    // re-arrange
    int buf;
    expect_token(ifp, SEPS, "<ReArrange>"); fscanf(ifp, "%d %d", &odim, &idim);
    assert(odim == band_points * nband);
    assert(idim == fbank_dim * nsplice * delta);
    expect_token(ifp, SEPS, "<PatchStrideIn>"); fscanf(ifp, "%d", &buf); assert(buf == fbank_dim);
    expect_token(ifp, SEPS, "<PatchDim>");      fscanf(ifp, "%d", &buf); assert(buf == band_width);
    expect_token(ifp, SEPS, "<PatchStep>");     fscanf(ifp, "%d", &buf); assert(buf == band_shift); 

    // parallel
    expect_token(ifp, SEPS, "<ParallelComponent>"); fscanf(ifp, "%d %d", &odim, &idim);
    int total_filter = odim;
    int nfilter = total_filter/nband;
    assert(idim == band_points * nband);
    expect_token(ifp, SEPS, "<NestedNnetCount>"); fscanf(ifp, "%d", &buf); assert(buf == nband);
    // range over subbands
    std::vector<Matrix> band_filters(nband);
    std::vector<std::vector<float> > band_bias(nband);

    int nr = nfilter;
    int nc = filter_size;
    for (int b = 0; b < nband; b++) {
        expect_token(ifp, SEPS, "<NestedNnet>"); fscanf(ifp, "%d", &buf); assert(buf == b+1);
        expect_token(ifp, SEPS, "<Nnet>");
        expect_token(ifp, SEPS, "<ConvolutionalComponent>"); fscanf(ifp, "%d %d", &odim, &idim); 
        assert(odim == nfilter * pool_size); assert(idim == band_points);
        expect_token(ifp, SEPS, "<PatchDim>"); fscanf(ifp, "%d", &buf); assert(buf == patch_dim);
        expect_token(ifp, SEPS, "<PatchStep>"); fscanf(ifp, "%d", &buf); assert(buf == patch_step);
        expect_token(ifp, SEPS, "<PatchStride>"); fscanf(ifp, "%d", &buf); assert(buf == patch_stride);

        // filters
        band_filters[b].Init(nr, nc);
        expect_token(ifp, SEPS, "<Filters>"); expect_token(ifp, SEPS, "[");
        for (int r = 0; r < nfilter; r++) {
            for (int c = 0; c < filter_size; c++) {
                fscanf(ifp, "%f ", &band_filters[b].data_[r * nc + c]);
            }
        }
        expect_token(ifp, SEPS, "]");

        // bias
        band_bias[b].resize(nfilter);
        expect_token(ifp, SEPS, "<Bias>"); expect_token(ifp, SEPS, "[");
        for (int i = 0; i < nfilter; i++) {
            fscanf(ifp, "%f ", &band_bias[b][i]);
        }
        expect_token(ifp, SEPS, "]");

        // max-pooling
        expect_token(ifp, SEPS, "<MaxPoolingComponent>"); fscanf(ifp, "%d %d", &odim, &idim);
        assert(odim == nfilter); assert(idim = nfilter * pool_size);
        expect_token(ifp, SEPS, "<PoolSize>"); fscanf(ifp, "%d ", &buf); assert(buf = pool_size);
        expect_token(ifp, SEPS, "<PoolStep>"); fscanf(ifp, "%d ", &buf); assert(buf = pool_size);
        expect_token(ifp, SEPS, "<PoolStride>"); fscanf(ifp, "%d ", &buf); assert(buf = nfilter);
        expect_token(ifp, SEPS, "</Nnet>");
    }
    expect_token(ifp, SEPS, "</ParallelComponent>");

//    // check filters
//    for (int b = 0; b < nband; b++) {
//        fprintf(stderr, "band %d filters\n", b);
//        for (int f = 0; f < nfilter; f++) {
//            for (int c = 0; c < filter_size; c++) {
//                fprintf(stderr, "%f ", band_filters[b].data_[f * nc + c]);
//            }
//            fprintf(stderr, "\n");
//        }
//    }
//    // check bias
//    for (int b = 0; b < nband; b++) {
//        fprintf(stderr, "band %d bias\n", b);
//        for (int f = 0; f < nfilter; f++) {
//            fprintf(stderr, "%f ", band_bias[b][f]);
//        }
//        fprintf(stderr, "\n");
//    }

    // sigmoid after conv+maxpooling
    expect_token(ifp, SEPS, "<Sigmoid>");
    fscanf(ifp, "%d %d", &odim, &idim);
    assert(odim == idim);
    assert(odim == total_filter);

    // following layers (hidden, sigmoid, softmax)
    std::vector<Matrix*> matps;
    matps.reserve(20);
    std::vector<std::vector<float> > biases;
    biases.reserve(20);

    while(peek_token(ifp, SEPS, tok)) {
        if (strcmp(tok, "<AffineTransform>") == 0) {
            expect_token(ifp, SEPS, "<AffineTransform>");
            fscanf(ifp, "%d %d", &odim, &idim);
            //printf("%d %d\n", odim, idim);
            //expect_token(ifp, SEPS, "<LearnRateCoef>"); fscanf(ifp, "%f ", &buf);
            //expect_token(ifp, SEPS, "<BiasLearnRateCoef>"); fscanf(ifp, "%f ", &buf);

            nr = odim;
            nc = idim;

            // hidden
            Matrix *matp = new Matrix;
            matps.push_back(matp);
            matp->Init(nr, nc);
            expect_token(ifp, SEPS, "[");
            for (int r = 0; r < nr; r++) {
                for (int c = 0; c < nc; c++) {
                    fscanf(ifp, "%f ", &matp->data_[r*nc + c]);
                }
            }
            expect_token(ifp, SEPS, "]");

            // bias
            biases.push_back(std::vector<float>());
            biases.back().resize(odim);
            expect_token(ifp, SEPS, "[");
            for (int i = 0; i < odim; i++) {
                fscanf(ifp, "%f ", &(biases.back()[i]));
            }
            expect_token(ifp, SEPS, "]");
        } else if (strcmp(tok, "<Sigmoid>") == 0) {
            expect_token(ifp, SEPS, "<Sigmoid>");
            fscanf(ifp, "%d %d", &odim, &idim);
        } else if (strcmp(tok, "<Softmax>") == 0) {
            expect_token(ifp, SEPS, "<Softmax>");
            fscanf(ifp, "%d %d", &odim, &idim);
        } else {
            break;
        }
    }
    expect_token(ifp, SEPS, "</Nnet>");

//    fprintf(stderr, "number of hidden components: %d\n", matps.size());
//
//    // check
//    for (int i = 0; i < matps.size(); i++) {
//        Matrix *matp = matps[i];
//        // hidden
//        fprintf(stderr, "hidden %d:\n", i);
//        for (int r = 0; r < matp->nr_; r++) {
//            for (int c = 0; c < matp->nc_; c++) {
//                fprintf(stderr, "%f ", matp->data_[r * matp->nc_ + c]);
//            }
//            fprintf(stderr, "\n");
//        }
//        // bias
//        fprintf(stderr, "bias %d:\n", i);
//        std::vector<float> &bias = biases[i];
//        for (int k = 0; k < bias.size(); k++) {
//            fprintf(stderr, "%f ", bias[k]);
//        }
//        fprintf(stderr, "\n");
//    }


    // output wmatrix
    // conv layer
    fwrite(&total_filter, sizeof(total_filter), 1, ofp);
    buf = filter_size + 1; // bias included
    fwrite(&buf, sizeof(buf), 1, ofp);
    for (int b = 0; b < nband; b++) {
        for (int i = 0; i < nfilter; i++) {
            float *coefs = &band_filters[b].data_[i * filter_size];
            std::vector<float> reorder(filter_size);
            int F = patch_dim;  // freq
            int D = delta;      // delta
            int T = nsplice;    // time axis splice
            int f, d, t;
            for (f = 0; f < F; ++f) {
                for (t = 0; t < T; ++t) {
                    for (d = 0; d < D; ++d) {
                        int dst = (f*T + t)*D + d;
                        int src = (t*D + d)*F + f;
                        reorder[dst] = coefs[src];
                    }
                }
            }
            fwrite(&reorder[0], sizeof(float), filter_size, ofp);
            fwrite(&band_bias[b][i], sizeof(float), 1, ofp);
        }
    }

    Matrix *matp;
    // first hidden layer needs reorder
    matp = matps[0];
    buf = matp->nc_+1;
    fwrite(&buf, sizeof(float), 1, ofp);
    buf = matp->nr_+1;
    fwrite(&buf, sizeof(float), 1, ofp);
    // linearity
    for (int i = 0; i < nfilter; i++) {
        for (int b = 0; b < nband; b++) {
            int c = b * nfilter + i;  // reorder
            for (int r = 0; r < matp->nr_; r++) {
                fwrite(&matp->data_[r * matp->nc_ + c], sizeof(float), 1, ofp);
            }
            // dummy
            float f = 0.0;
            fwrite(&f, sizeof(float), 1, ofp);
        }
    }
    // bias
    fwrite(&biases[0][0], sizeof(float), biases[0].size(), ofp);
    // dummy
    float f = 0.0;
    fwrite(&f, sizeof(float), 1, ofp);

    // normal hidden layers
    for (int i = 1; i < matps.size()-1; i++) {
        // linearity
        matp = matps[i];
        buf = matp->nc_+1;
        fwrite(&buf, sizeof(float), 1, ofp);
        buf = matp->nr_+1;
        fwrite(&buf, sizeof(float), 1, ofp);
        for (int c = 0; c < matp->nc_; c++) {
            for (int r = 0; r < matp->nr_; r++) {
                fwrite(&matp->data_[r * matp->nc_ + c], sizeof(float), 1, ofp);
            }
            // dummy
            float f = 0.0;
            fwrite(&f, sizeof(float), 1, ofp);
        }
        // bias
        std::vector<float> &bias = biases[i];
        fwrite(&bias[0], sizeof(float), bias.size(), ofp);
        // dummy
        float f = 0.0;
        fwrite(&f, sizeof(float), 1, ofp);
    }

    // softmax layer do not have dummy nodes
    //   lineary
    matp = matps[matps.size()-1];
    buf = matp->nc_+1;
    fwrite(&buf, sizeof(float), 1, ofp);
    buf = matp->nr_;
    fwrite(&buf, sizeof(float), 1, ofp);
    for (int c = 0; c < matp->nc_; c++) {
        for (int r = 0; r < matp->nr_; r++) {
            fwrite(&matp->data_[r * matp->nc_ + c], sizeof(float), 1, ofp);
        }
    }
    //   bias
    std::vector<float> &bias = biases[biases.size()-1];
    fwrite(&biases[biases.size()-1][0], sizeof(float), biases[biases.size()-1].size(), ofp);

  // free hidden layer matrices
    for (int i = 0; i < matps.size(); i++) {
        delete matps[i];
    }

    fclose(ifp);
//    fclose(ofp);
    return EXIT_SUCCESS;
}

