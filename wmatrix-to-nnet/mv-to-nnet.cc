#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <cstring>

#include <vector>

int main(int argc, char ** argv) 
{
    if (argc != 3) {
        fprintf(stderr, "Usage: mv-to-nnet (i)global_mean_var (o)kaldi.norm.nnet.txt\n");
        exit(0);
    }
    char *meanvar_filename = argv[1];
    char *ofilename  = argv[2];

    FILE *ifp  = fopen(meanvar_filename, "r");         assert(ifp != NULL);
    FILE *ofp  = fopen(ofilename, "w+");               assert(ofp != NULL);

    // dims
    int nsplice = 11;   // seems to be stable config, fix it
    int delta = 3;
    int fbank_dim = 40;
    int idim, odim;

    // nnet head tag
    fprintf(ofp, "<Nnet> \n");

    // splice component
    idim = fbank_dim * delta;
    odim = fbank_dim * nsplice * delta;
    fprintf(ofp, "<Splice> %d %d\n", odim, idim);
    switch (nsplice) {
        case 15:
            fprintf(ofp, "[ -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7]\n"); break;
        case 13:
            fprintf(ofp, "[ -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6]\n"); break;
        case 11:
            fprintf(ofp, "[ -5 -4 -3 -2 -1 0 1 2 3 4 5 ]\n"); break;
        case 9:
            fprintf(ofp, "[ -4 -3 -2 -1 0 1 2 3 4 ]\n"); break;
        default:
            fprintf(stderr, "ERROR: nsplice %d is not supported.\n", nsplice);
            exit(0);
    }

    // load global mean and variance
    char line[256];
    std::vector<float> shift;
    std::vector<float> scale;
    while (fgets(line, 255, ifp) != NULL) {
        if (strcmp(line, "\n") == 0) continue;
        float mean = atof(strtok(line, " \t\n"));
        float var  = atof(strtok(NULL, " \t\n"));
        shift.push_back(-mean);
        scale.push_back(1/sqrt(var));
    }
    fprintf(stderr, "%d\n", shift.size());
    fclose(ifp);

    // cmn component
    idim = odim = nsplice * delta * fbank_dim;
    fprintf(ofp, "<AddShift> %d %d\n", odim, idim);
    fprintf(ofp, " [ ");
    for (int s = 0; s < nsplice; ++s) {
        for (int i = 0; i < shift.size(); ++i) {
            fprintf(ofp, "%f ", shift[i]);
        }
    }
    fprintf(ofp, " ] \n");

    // cvn component
    idim = odim = nsplice * delta * fbank_dim;
    fprintf(ofp, "<Rescale> %d %d\n", odim, idim);
    fprintf(ofp, " [ ");
    for (int s = 0; s < nsplice; ++s) {
        for (int i = 0; i < scale.size(); ++i) {
            fprintf(ofp, "%f ", scale[i]);
        }
    }
    fprintf(ofp, " ] \n");

    fprintf(ofp, "</Nnet> \n");

    return EXIT_SUCCESS;
}
