#ifndef FILE_UTILS
#define FILE_UTILS
/* 
 * read_token reads a token from a file
 * use characters in seps as token seperator
 * the space of token should be pre-allocated
 * the length of token is returned
 */
int read_token(FILE *fp, const char *seps, char *token);

/*
 * peek_token is the same as read_token except it doesn't consume the file stream
 */
int peek_token(FILE *fp, const char *seps, char *token);

/*
 * expect_token reads a token from the file and abort if it is different from expect
 */
void expect_token(FILE *fp, const char *seps, const char *expect);

#endif
