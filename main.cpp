#include <mpi.h>
#include <omp.h>
#include <getopt.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>


// processes per coordinate
int param_x_proc = 1;
int param_y_proc = 1;
int param_z_proc = 1;
// space-time boundaries
double param_x_bound = 1;
double param_y_bound = 1;
double param_z_bound = 1;
double param_t_bound = 1;
// quantification parameters
int param_x_nodes = 128;
int param_y_nodes = 128;
int param_z_nodes = 128;
int param_t_steps = 20;
// threads number 
int param_nt = 1;

// total number of processes and current process rank
int nproc, rank;
const int MASTER_PROCESS = 0;

typedef enum {
    SUCCESS            = 0,
    INVALID_PARAMETERS = 1
} exit_code;


bool validate_parameters() {

    bool is_valid = true;
    // TODO check for two-letter one-dash keys allowance

    if (param_x_proc <= 0) {
        std::cerr << "input parameters error: "
            "--xproc must be a positive integer, but " <<
            param_x_proc << " was given" << std::endl;
        is_valid = false;
    }

    if (param_y_proc <= 0) {
        std::cerr << "input parameters error: "
            "--yproc must be a positive integer, but " <<
            param_y_proc << " was given" << std::endl;
        is_valid = false;
    }

    if (param_z_proc <= 0) {
        std::cerr << "input parameters error: "
            "--zproc must be a positive integer, but " <<
            param_z_proc << " was given" << std::endl;
        is_valid = false;
    }

    if (param_x_bound <= 0) {
        std::cerr << "input parameters error: "
            "--xbound (-x) must be a positive double, but " <<
            param_x_bound << " was given" << std::endl;
        is_valid = false;
    }

    if (param_y_bound <= 0) {
        std::cerr << "input parameters error: "
            "--ybound (-y) must be a positive double, but " <<
            param_y_bound << " was given" << std::endl;
        is_valid = false;
    }

    if (param_z_bound <= 0) {
        std::cerr << "input parameters error: "
            "--zbound (-z) must be a positive double, but " <<
            param_z_bound << " was given" << std::endl;
        is_valid = false;
    }

    if (param_t_bound <= 0) {
        std::cerr << "input parameters error: "
            "--tbound (-t) must be a positive double, but " <<
            param_t_bound << " was given" << std::endl;
        is_valid = false;
    }

    if (param_x_nodes <= 0) {
        std::cerr << "input parameters error: "
            "--xnodes must be a positive integer, but " <<
            param_x_nodes << " was given" << std::endl;
        is_valid = false;
    }

    if (param_y_nodes <= 0) {
        std::cerr << "input parameters error: "
            "--ynodes must be a positive integer, but " <<
            param_y_nodes << " was given" << std::endl;
        is_valid = false;
    }

    if (param_z_nodes <= 0) {
        std::cerr << "input parameters error: "
            "--znodes must be a positive integer, but " <<
            param_z_nodes << " was given" << std::endl;
        is_valid = false;
    }

    if (param_t_steps <= 0) {
        std::cerr << "input parameters error: "
            "--tsteps (-s) must be a positive integer, but " <<
            param_t_steps << " was given" << std::endl;
        is_valid = false;
    }

    if (param_nt <= 0) {
        std::cerr << "input parameters error: "
            "--nt (-n) must a be positive integer, but " <<
            param_nt << " was given" << std::endl;
        is_valid = false;
    }

    return is_valid;
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // arrays for parameters transition from the master to slaves
    int params_int[8];
    double params_double[4];

    // processing and validating input values
    if (rank == MASTER_PROCESS) {

        int option_char = -1, 
            option_idx = 0;
        std::string help_message = 
            "Usage: <binary> [OPTIONS], where OPTIONS include\n"
            "[--help | -h] \n"
            "--xproc=<number of processes for X axis of computational grid> \n"
            "--yproc=<number of processes for Y axis of computational grid> \n"
            "--zproc=<number of processes for Z axis of computational grid> \n"
            "[--xbound=<upper X boundary for computational grid> | -x <...>] \n"
            "[--ybound=<upper Y boundary for computational grid> | -y <...>] \n"
            "[--zbound=<upper Z boundary for computational grid> | -z <...>] \n"
            "[--tbound=<temporal upper limit for the model> | -t <...>] \n"
            "--xnodes=<number of grid nodes per X axis> \n"
            "--ynodes=<number of grid nodes per Y axis> \n"
            "--znodes=<number of grid nodes per Z axis> \n"
            "[--tsteps=<number of iterations to perform> | -s <...>] \n"
            "[--nt=<thread number> | -n <...>] \n";

        while (true) {

            static struct option long_options[] = {
                { "xproc",  optional_argument, 0,  1  },
                { "yproc",  optional_argument, 0,  2  },
                { "zproc",  optional_argument, 0,  3  },
                { "xnodes", optional_argument, 0,  4  },
                { "ynodes", optional_argument, 0,  5  },
                { "znodes", optional_argument, 0,  6  },
                { "xbound", optional_argument, 0, 'x' },
                { "ybound", optional_argument, 0, 'y' },
                { "zbound", optional_argument, 0, 'z' },
                { "tbound", optional_argument, 0, 't' },
                { "tsteps", optional_argument, 0, 's' },
                { "nt",     optional_argument, 0, 'n' },
                { "help",   no_argument,       0, 'h' },
                { 0,        0,                 0,  0  }
            };

            option_char = getopt_long(argc, argv, "x:y:z:t:s:n:h", 
                long_options, &option_idx);
            if (option_char == -1) {
                break;
            }

            switch (option_char) {
                case 1:
                    param_x_proc = atoi(optarg);
                    break;

                case 2:
                    param_y_proc = atoi(optarg);
                    break;

                case 3:
                    param_z_proc = atoi(optarg);
                    break;

                case 4:
                    param_x_nodes = atoi(optarg);
                    break;

                case 5:
                    param_y_nodes = atoi(optarg);
                    break;

                case 6:
                    param_z_nodes = atoi(optarg);
                    break;

                case 'x':
                    param_x_bound = atof(optarg);
                    break;

                case 'y':
                    param_y_bound = atof(optarg);
                    break;

                case 'z':
                    param_z_bound = atof(optarg);
                    break;

                case 't':
                    param_t_bound = atof(optarg);
                    break;

                case 's':
                    param_t_steps = atoi(optarg);
                    break;

                case 'n':
                    param_nt = atoi(optarg);
                    break;

                case 'h':
                    std::cout << help_message << std::endl;
                    break;

                case '?':
                    break;
            }
        }

        if (!validate_parameters()) {
            std::cout << help_message << std::endl;
            MPI_Abort(MPI_COMM_WORLD, INVALID_PARAMETERS);
            exit(INVALID_PARAMETERS);
        }

        params_int[0] = param_x_proc;
        params_int[1] = param_y_proc;
        params_int[2] = param_z_proc;
        params_int[3] = param_x_nodes;
        params_int[4] = param_y_nodes;
        params_int[5] = param_z_nodes;
        params_int[6] = param_t_steps;
        params_int[7] = param_nt;

        params_double[0] = param_x_bound;
        params_double[1] = param_y_bound;
        params_double[2] = param_z_bound;
        params_double[3] = param_t_bound;
    }

    // receiving parameters from MASTER
    MPI_Bcast(params_int, 8, MPI_INT, MASTER_PROCESS, MPI_COMM_WORLD);
    MPI_Bcast(params_double, 4, MPI_DOUBLE, MASTER_PROCESS, MPI_COMM_WORLD);

    param_x_proc  = params_int[0];
    param_y_proc  = params_int[1];
    param_z_proc  = params_int[2];
    param_x_nodes = params_int[3];
    param_y_nodes = params_int[4];
    param_z_nodes = params_int[5];
    param_t_steps = params_int[6];
    param_nt      = params_int[7];

    param_x_bound = params_double[0];
    param_y_bound = params_double[1];
    param_z_bound = params_double[2];
    param_t_bound = params_double[3];

    std::cout << "param_x_proc=" << param_x_proc << std::endl <<
                 "param_y_proc=" << param_y_proc << std::endl <<
                 "param_z_proc=" << param_z_proc << std::endl <<
                 "param_x_bound=" << param_x_bound << std::endl <<
                 "param_y_bound=" << param_y_bound << std::endl <<
                 "param_z_bound=" << param_z_bound << std::endl <<
                 "param_t_bound=" << param_t_bound << std::endl <<
                 "param_x_nodes=" << param_x_nodes << std::endl <<
                 "param_y_nodes=" << param_y_nodes << std::endl <<
                 "param_z_nodes=" << param_z_nodes << std::endl <<
                 "param_t_steps=" << param_t_steps << std::endl <<
                 "param_nt=" << param_nt << std::endl <<
                 "in rank #" << rank << std::endl;

    omp_set_num_threads(param_nt);
    /*
        GRID MANIPULATION AND CALCULATIONS
    */

    MPI_Finalize();
    exit(SUCCESS);
}
