#include <cstdio>
#include <cstring>
#include <cassert>

#include "file-utils.h"

int read_token(FILE *fp, const char *seps, char *token)
{
    int len = 0;
    int ch;
    while((ch = fgetc(fp)) != EOF) {
        if (strchr(seps, ch)) {
            if (len != 0) break;
        } else {
            token[len++] = ch;
        }
    }
    token[len] = '\0';
    return len;
}

int peek_token(FILE *fp, const char *seps, char *token)
{
    int len;
    fpos_t pos;
    fgetpos(fp, &pos);
    len = read_token(fp, seps, token);
    fsetpos(fp, &pos);
    return len;
}

void expect_token(FILE *fp, const char *seps, const char *expect)
{
    char tok[256];
    int n = 0;
    n = read_token(fp, seps, tok);
    assert(n < 256);
    assert(strcmp(tok, expect) == 0);
}

