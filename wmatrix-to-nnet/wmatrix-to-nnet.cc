#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <cstring>

#include <vector>

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
    int ReadFile(FILE *fp);
    int WriteFile(FILE *fp, int last);
};

int Matrix::ReadFile(FILE *fp) {
    fread(&nr_, sizeof(nr_), 1, fp);
    fread(&nc_, sizeof(nc_), 1, fp);
    data_ = (float*)malloc(nr_ * nc_ * sizeof(data_[0]));
    fread(data_, sizeof(data_[0]), nr_ * nc_, fp);
    return 0;
}

int Matrix::WriteFile(FILE *fp, int hidden) {
    // last row in WMatrix is bias, write out latter
    // last col in WMatrix is dummy
    int idim = nr_-1;
    int odim = hidden ? nc_-1 : nc_;  // hidden layer has a tailed col for dummy-bias
    fprintf(fp, "<AffineTransform> %d %d\n", odim, idim);
    //fprintf(fp, "<LearnRateCoef> 1 <BiasLearnRateCoef> 1 \n");

    // linearity
    fprintf(fp, " [\n");
    for (int c = 0; c < odim; ++c) {  
        fprintf(fp, "  ");
        for (int r = 0; r < idim; ++r) {  
            fprintf(fp, "%f ", data_[r*nc_ + c]);
        }
        fprintf(fp, " \n");
    }
    fprintf(fp, " ] \n");

    // bias
    fprintf(fp, " [ ");
    for (int c = 0; c < odim; ++c) {
        fprintf(fp, "%f ", data_[(nr_-1)*nc_ + c]);
    }
    fprintf(fp, " ] \n");
}

size_t GetEofPosition(FILE* fp) {
    rewind(fp);
    fseek(fp, 0, SEEK_END);
    size_t eofpos = ftell(fp);
    rewind(fp);
    return eofpos;
}

void filter_coefs_reorder(const std::vector<float> &coefs, std::vector<float> &reorder, int patch_dim, int delta, int nsplice) {
    int F = patch_dim;  // freq
    int D = delta;      // delta
    int T = nsplice;    // time axis splice
    int f, d, t;
    for (f = 0; f < F; ++f) {
        for (t = 0; t < T; ++t) {
            for (d = 0; d < D; ++d) {
                int src = (f*T + t)*D + d;
                int dst = (t*D + d)*F + f;
                reorder[dst] = coefs[src];
            }
        }
    }
}

int main(int argc, char ** argv) 
{
    if (argc != 3) {
        fprintf(stderr, "Usage: wmatrix-to-nnet (i)WMatrix (o)kaldi.nnet.txt\n");
        exit(0);
    }


    char *imodel_filename  = argv[1];
    //char *meanvar_filename = argv[2];
    char *omodel_filename  = argv[2];

    FILE *ifp  = fopen(imodel_filename, "r");         assert(ifp != NULL);
    //FILE *meanvar_fp = fopen(meanvar_filename, "r");  assert(meanvar_fp != NULL);
    FILE *ofp  = fopen(omodel_filename, "w+");        assert(ofp != NULL);

    int idim, odim;

    // load WMatrix
    size_t eofpos = GetEofPosition(ifp);
    std::vector<Matrix*> wmatrix;
    int layer = 1;
    fprintf(stderr, "WMatrix dimensions:\n");
    while (ftell(ifp) < eofpos) {
        Matrix *matp = new Matrix;
        matp->ReadFile(ifp);
        wmatrix.push_back(matp);
        fprintf(stderr, "    WMatrix Layer%d: %d rows, %d cols\n", layer, matp->nr_, matp->nc_);
        ++layer;
    }
    fprintf(stderr, "\n");
    fclose(ifp);

//    // load global mean and variance
//    char line[256];
//    std::vector<float> shift;
//    std::vector<float> scale;
//    while (fgets(line, 255, meanvar_fp) != NULL) {
//        if (strcmp(line, "\n") == 0) continue;
//        float mean = atof(strtok(line, " \t\n"));
//        float var  = atof(strtok(NULL, " \t\n"));
//        shift.push_back(-mean);
//        scale.push_back(1/sqrt(var));
//    }
//    fprintf(stderr, "%d\n", shift.size());
//    fclose(meanvar_fp);

    // useful dims and logging
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

    int total_filter = wmatrix[0]->nr_;             // first matrix is convolutional, each row for a filter
    int nfilter = total_filter / nband;             // nfilters per subband
    assert(wmatrix[0]->nc_ == filter_size + 1);     // each row is a filter coefs + 1 bias

    fprintf(stderr, "fbank dim: %d \n", fbank_dim);
    fprintf(stderr, "delta: %d \n", delta);
    fprintf(stderr, "num spliced: %d \n\n", nsplice);

    fprintf(stderr, "num of filters in total: %d\n", total_filter);
    fprintf(stderr, "num of subbands: %d\n", nband);
    fprintf(stderr, "num of filters per subband: %d\n\n", nfilter);
    fprintf(stderr, "subband points: freq-axis(%d) * delta(%d) * time-axis(%d) = %d \n", 
            band_width, 
            delta, 
            nsplice, 
            band_points);
    fprintf(stderr, "filter  points: freq-axis(%d) * delta(%d) * time-axis(%d) = %d (without bias)\n", 
            patch_dim, 
            delta,
            nsplice, 
            filter_size);
    fprintf(stderr, "filter step along freq-axis: %d \n", patch_step);
    fprintf(stderr, "pool size: %d \n", pool_size);


    // nnet head tag
    fprintf(ofp, "<Nnet> \n");

//    // splice component
//    idim = fbank_dim * delta;
//    odim = fbank_dim * nsplice * delta;
//    fprintf(ofp, "<Splice> %d %d\n", odim, idim);
//    switch (nsplice) {
//        case 15:
//            fprintf(ofp, "[ -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7]\n"); break;
//        case 13:
//            fprintf(ofp, "[ -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6]\n"); break;
//        case 11:
//            fprintf(ofp, "[ -5 -4 -3 -2 -1 0 1 2 3 4 5 ]\n"); break;
//        case 9:
//            fprintf(ofp, "[ -4 -3 -2 -1 0 1 2 3 4 ]\n"); break;
//        default:
//            fprintf(stderr, "ERROR: nsplice %d is not supported.\n", nsplice);
//            exit(0);
//    }
//
//    // cmn component
//    idim = odim = nsplice * delta * fbank_dim;
//    fprintf(ofp, "<AddShift> %d %d\n", odim, idim);
//    fprintf(ofp, " [ ");
//    for (int s = 0; s < nsplice; ++s) {
//        for (int i = 0; i < shift.size(); ++i) {
//            fprintf(ofp, "%f ", shift[i]);
//        }
//    }
//    fprintf(ofp, " ] \n");
//
//    // cvn component
//    idim = odim = nsplice * delta * fbank_dim;
//    fprintf(ofp, "<Rescale> %d %d\n", odim, idim);
//    fprintf(ofp, " [ ");
//    for (int s = 0; s < nsplice; ++s) {
//        for (int i = 0; i < scale.size(); ++i) {
//            fprintf(ofp, "%f ", scale[i]);
//        }
//    }
//    fprintf(ofp, " ] \n");

    // re-arrange component
    idim = fbank_dim * delta * nsplice;
    odim = band_width * nband * delta * nsplice;
    fprintf(ofp, "<ReArrange> %d %d \n", odim, idim);
    fprintf(ofp, "<PatchStrideIn> %d <PatchDim> %d <PatchStep> %d \n", fbank_dim, band_width, band_shift);
    
    // parallel component, one conv+pooling nnet (nested in parallel) for each subband
    Matrix *matp = wmatrix[0];
    fprintf(ofp, "<ParallelComponent> %d %d \n", 
            total_filter, 
            band_width * delta * nsplice * nband);
    fprintf(ofp, "<NestedNnetCount> %d \n", nband);

    // range over subbands (nested nnet in parallel component)
    for (int b = 0; b != nband; ++b) {
        fprintf(ofp, "<NestedNnet> %d ", b+1);
        fprintf(ofp, "<Nnet>\n");

        // convolutional component
        fprintf(ofp, "<ConvolutionalComponent> %d %d \n", 
                pool_size * nfilter, 
                band_width * delta * nsplice);
        fprintf(ofp, "<PatchDim> %d <PatchStep> %d <PatchStride> %d <Filters> ", 
                patch_dim,
                patch_step,
                patch_stride);

        // range over subband filters
        fprintf(ofp, " [ \n");
        std::vector<float> bias;
        for (int f = 0; f != nfilter; ++f) {
            fprintf(ofp, "  ");
            // reorder filter coefs
            std::vector<float> coefs(filter_size);
            
            int row = b * nfilter + f;

            float *rowp = &(matp->data_[row * matp->nc_]);
            // get raw filter coefs
            int i;
            for (i = 0; i < filter_size ; ++i) {
                coefs[i] = rowp[i];
            }
            bias.push_back(rowp[i]); // last col is bias

            // reorder filter coefs
            std::vector<float> reorder(filter_size);
            filter_coefs_reorder(coefs, reorder, patch_dim, delta, nsplice);

            // write out reordered filter coefs
            for (int k = 0; k < filter_size; ++k) {
                fprintf(ofp, "%f ", reorder[k]);
            }

            fprintf(ofp, "\n");
        }
        fprintf(ofp, " ] \n");

        // filter biases
        assert(bias.size() == nfilter);
        fprintf(ofp, "<Bias> [ ");
        for (int k = 0; k < nfilter; ++k) {
            fprintf(ofp, "%f ", bias[k]);
        }
        fprintf(ofp, " ] \n");
        
        // max-pooling component
        fprintf(ofp, "<MaxPoolingComponent> %d %d \n", nfilter, nfilter * pool_size);
        fprintf(ofp, "<PoolSize> %d <PoolStep> %d <PoolStride> %d ", pool_size, pool_size, nfilter);
        fprintf(ofp, "</Nnet>\n");
    }
    fprintf(ofp, "</ParallelComponent> \n");

    // sigmoid after pooling
    idim = odim = total_filter;
    fprintf(ofp, "<Sigmoid> %d %d \n", odim, idim);

    // reorder weights between max-pooling output and next sigmoid layer
    matp = wmatrix[1];

    idim = matp->nr_-1;
    odim = matp->nc_-1;
    fprintf(ofp, "<AffineTransform> %d %d\n", odim, idim);
    //fprintf(ofp, "<LearnRateCoef> 1 <BiasLearnRateCoef> 1 \n");

    fprintf(ofp, " [ \n");
    for (int c = 0; c < odim; ++c) {
        fprintf(ofp, "  ");
        for (int b = 0; b < nband; ++b) {
            for (int f = 0; f < nfilter; ++f) {
                int row = f * nband + b;
                fprintf(ofp, "%f ", matp->data_[row * matp->nc_ + c]);
            }
        }
        fprintf(ofp, "\n");
    }
    fprintf(ofp, " ] \n");

    fprintf(ofp, " [ ");

    int r = matp->nr_-1;  // bias (last row)
    for (int c = 0; c < odim; ++c) {
        fprintf(ofp, "%f ", matp->data_[r * matp->nc_ + c]);
    }
    fprintf(ofp, " ] \n");

    idim = odim;
    fprintf(ofp, "<Sigmoid> %d %d \n", odim, idim);

    // higher hidden layers
    for (int i = 2; i < wmatrix.size()-1; ++i) {
        matp = wmatrix[i];
        int hidden = 1;
        // linearity + bias
        matp->WriteFile(ofp, hidden);
        // sigmoid
        idim = odim = matp->nc_ - 1;    // "- 1" get rid of dummy-bias-neuron
        fprintf(ofp, "<Sigmoid> %d %d \n", odim, idim);
    }

    // last softmax layer
    matp = wmatrix.back();
    int hidden = 0;
    matp->WriteFile(ofp, hidden);
    idim = odim = matp->nc_;    // softmax doesn't have dummy-bias-neuron, no need to -1
    fprintf(ofp, "<Softmax> %d %d \n", odim, idim);

    // nnet end tag
    fprintf(ofp, "</Nnet> \n");

    fclose(ofp);

    // free up
    for (int i = 0; i < wmatrix.size(); ++i) {
        delete(wmatrix[i]);
    }

    return EXIT_SUCCESS;
}
