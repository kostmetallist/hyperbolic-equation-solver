#define _USE_MATH_DEFINES

#include <mpi.h>
#include <omp.h>
#include <getopt.h>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <math.h>
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
    SUCCESS                 = 0,
    INVALID_PARAMETERS      = 1,
    PROCESS_NUMBER_MISMATCH = 2
} exit_code;

typedef struct {
    int from;
    int to;
} Range;

typedef struct {
    Range x;
    Range y;
    Range z;
} ProcessBlockArea;

typedef struct {
    ProcessBlockArea inner;
    // extended area represents the union of inner cells and interface ones
    ProcessBlockArea extended;
} ProcessBlock;

class MatrixAccessor {
private:
    double *data;
    int xn, yn, zn;

public:

    MatrixAccessor(double *data, int xn, int yn, int zn) {
        this->data = data;
        this->xn = xn;
        this->yn = yn;
        this->zn = zn;
    }

    double get(int x, int y, int z) const {
        return this->data[x * this->zn * this->yn + y * this->yn + z];
    }

    void set(int x, int y, int z, double value) {
        this->data[x * this->zn * this->yn + y * this->yn + z] = value;
    }
};


exit_code validate_parameters(int claimed_process_number) {

    exit_code result = SUCCESS;

    if (param_x_proc <= 0) {
        std::cerr << "input parameters error: "
            "--xproc must be a positive integer, but " <<
            param_x_proc << " was given" << std::endl;
        result = INVALID_PARAMETERS;
    }

    if (param_y_proc <= 0) {
        std::cerr << "input parameters error: "
            "--yproc must be a positive integer, but " <<
            param_y_proc << " was given" << std::endl;
        result = INVALID_PARAMETERS;
    }

    if (param_z_proc <= 0) {
        std::cerr << "input parameters error: "
            "--zproc must be a positive integer, but " <<
            param_z_proc << " was given" << std::endl;
        result = INVALID_PARAMETERS;
    }

    if (param_x_bound <= 0) {
        std::cerr << "input parameters error: "
            "--xbound (-x) must be a positive double, but " <<
            param_x_bound << " was given" << std::endl;
        result = INVALID_PARAMETERS;
    }

    if (param_y_bound <= 0) {
        std::cerr << "input parameters error: "
            "--ybound (-y) must be a positive double, but " <<
            param_y_bound << " was given" << std::endl;
        result = INVALID_PARAMETERS;
    }

    if (param_z_bound <= 0) {
        std::cerr << "input parameters error: "
            "--zbound (-z) must be a positive double, but " <<
            param_z_bound << " was given" << std::endl;
        result = INVALID_PARAMETERS;
    }

    if (param_t_bound <= 0) {
        std::cerr << "input parameters error: "
            "--tbound (-t) must be a positive double, but " <<
            param_t_bound << " was given" << std::endl;
        result = INVALID_PARAMETERS;
    }

    if (param_x_nodes <= 0) {
        std::cerr << "input parameters error: "
            "--xnodes must be a positive integer, but " <<
            param_x_nodes << " was given" << std::endl;
        result = INVALID_PARAMETERS;
    }

    if (param_y_nodes <= 0) {
        std::cerr << "input parameters error: "
            "--ynodes must be a positive integer, but " <<
            param_y_nodes << " was given" << std::endl;
        result = INVALID_PARAMETERS;
    }

    if (param_z_nodes <= 0) {
        std::cerr << "input parameters error: "
            "--znodes must be a positive integer, but " <<
            param_z_nodes << " was given" << std::endl;
        result = INVALID_PARAMETERS;
    }

    if (param_t_steps <= 0) {
        std::cerr << "input parameters error: "
            "--tsteps (-s) must be a positive integer, but " <<
            param_t_steps << " was given" << std::endl;
        result = INVALID_PARAMETERS;
    }

    if (param_nt <= 0) {
        std::cerr << "input parameters error: "
            "--nt (-n) must a be positive integer, but " <<
            param_nt << " was given" << std::endl;
        result = INVALID_PARAMETERS;
    }

    int grid_process_number = param_x_proc * param_y_proc * param_z_proc;
    if (grid_process_number != claimed_process_number) {
        std::cerr << "process number mismatch: " << claimed_process_number <<
            " processes have been dedicated for the execution, but " <<
            grid_process_number << " have been specified in the parameters" <<
            std::endl;
        result = PROCESS_NUMBER_MISMATCH;
    }

    return result;
}

ProcessBlock prepare_process_block(const int axis_nodes[3],
    const int axis_coordinates[3], const int total_axis_processes[3]) {

    ProcessBlock block;
    ProcessBlockArea inner, extended;
    Range inner_ranges[3], extended_ranges[3];

    for (int i = 0; i < 3; ++i) {
        int raw_from = axis_coordinates[i] *
            (axis_nodes[i] / total_axis_processes[i]) + std::min(
                axis_coordinates[i], axis_nodes[i] % total_axis_processes[i]);
        Range inner_range, extended_range;

        inner_range.from = (raw_from > 0) ? raw_from: raw_from + 1;
        inner_range.to = raw_from + axis_nodes[i] / total_axis_processes[i] +
            ((axis_coordinates[i] < axis_nodes[i] % total_axis_processes[i]) ? 1: 0);
        inner_ranges[i] = inner_range;

        extended_range.from = (raw_from > 0) ? raw_from - 1: raw_from;
        extended_range.to = inner_range.to + 1;
        extended_ranges[i] = extended_range;
    }

    inner.x = inner_ranges[0];
    inner.y = inner_ranges[1];
    inner.z = inner_ranges[2];

    extended.x = extended_ranges[0];
    extended.y = extended_ranges[1];
    extended.z = extended_ranges[2];

    block.inner = inner;
    block.extended = extended;
    return block;
}

inline double sqr(double x) {
    return x * x;
}

double u_analytical(double x, double y, double z, double t) {
    return sin(M_PI * x / param_x_bound) * 
           sin(2 * M_PI * y / param_y_bound) *
           sin(3 * M_PI * z / param_z_bound) *
           cos(M_PI * t * sqrt(1 / sqr(param_x_bound) + 
                               4 / sqr(param_y_bound) + 
                               9 / sqr(param_z_bound)));
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

        exit_code validation_verdict = validate_parameters(nproc);
        if (validation_verdict != SUCCESS) {
            std::cout << help_message << std::endl;
            MPI_Abort(MPI_COMM_WORLD, validation_verdict);
            exit(validation_verdict);
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

    // runtime-calculated parameters
    const double dx = param_x_bound / param_x_nodes;
    const double dy = param_y_bound / param_y_nodes;
    const double dz = param_z_bound / param_z_nodes;

    // dt (time step) needs to be < min(dx, dy, dz) for calculation method to be stable
    const double dt = std::min(dx, std::min(dy, dz)) / 2;
    // TODO: uncomment for production
    // const double dt = param_t_bound / param_t_steps;

    const int dims[3] = {param_x_proc, param_y_proc, param_z_proc};
    const int periods[3] = {0, 0, 0};
    const int reorder_enabled = 1;
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder_enabled, &grid_comm);
    MPI_Comm_rank(grid_comm, &rank);
    int rank_coords[3];
    MPI_Cart_coords(grid_comm, rank, 3, rank_coords);

    // TODO: change placement to somewhat more appropriate
    omp_set_num_threads(param_nt);

    const int nodes_coupled[3] = {param_x_nodes, param_y_nodes, param_z_nodes};
    ProcessBlock pb = prepare_process_block(nodes_coupled, rank_coords, dims);

    const int block_cells[3] = {
        pb.extended.x.to - pb.extended.x.from,
        pb.extended.y.to - pb.extended.y.from,
        pb.extended.z.to - pb.extended.z.from
    };

    double *u_calc_prev = new double[block_cells[0] * block_cells[1] * block_cells[2]];
    double *u_calc_curr = new double[block_cells[0] * block_cells[1] * block_cells[2]];
    double *u_calc_next = new double[block_cells[0] * block_cells[1] * block_cells[2]];

    MatrixAccessor prev_accessor(u_calc_prev, block_cells[0], block_cells[1], block_cells[2]),
                   curr_accessor(u_calc_curr, block_cells[0], block_cells[1], block_cells[2]),
                   next_accessor(u_calc_next, block_cells[0], block_cells[1], block_cells[2]);

    // every block in non-trivial case has 6 neighbors: left-right, front-back, top-bottom
    const int interface_size = 6 * sqr(std::max(
        block_cells[0], std::max(block_cells[1], block_cells[2])));

    double *egress_buffer = new double[interface_size],
          *ingress_buffer = new double[interface_size];


    delete [] u_calc_prev;
    delete [] u_calc_curr;
    delete [] u_calc_next;

    delete [] egress_buffer;
    delete [] ingress_buffer;

    MPI_Finalize();
    exit(SUCCESS);
}
