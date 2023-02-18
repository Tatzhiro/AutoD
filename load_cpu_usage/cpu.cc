#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#define INTERVAL    500000
volatile sig_atomic_t   flag;
void setflag(int sig) { flag = 1; }

int main(int ac, char **av) {
    int load = 80;
    struct sigaction sigact;
    struct itimerval interval = { { 0, INTERVAL }, { 0, INTERVAL } };
    struct timespec pausetime = { 0, 0 };
    memset(&sigact, 0, sizeof(sigact));
    sigact.sa_handler = setflag;
    sigaction(SIGALRM, &sigact, 0);
    setitimer(ITIMER_REAL, &interval, 0);
    if (ac == 2) load = atoi(av[1]);
    pausetime.tv_nsec = INTERVAL*(100 - load)*10;
    while (1) {
        flag = 0;
        nanosleep(&pausetime, 0);
        while (!flag) { /* spin */ } }
    return 0;
}