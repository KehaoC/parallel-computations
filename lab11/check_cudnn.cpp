#include <cudnn.h>
#include <stdio.h>

int main() {
    printf("cuDNN version: %d\n", CUDNN_VERSION);
    return 0;
}