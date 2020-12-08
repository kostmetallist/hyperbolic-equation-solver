#define _USE_MATH_DEFINES

#include <mpi.h>
#include <omp.h>
#include <getopt.h>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <stdlib.h>

#include "structures.hpp"


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

// MPI shift dispositions
const int DISP_UPWARDS = 1;
const int DISP_DOWNWARDS = -1;

typedef enum {
    SUCCESS                 = 0,
    INVALID_PARAMETERS      = 1,
    PROCESS_NUMBER_MISMATCH = 2
} exit_code;


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

double calculate_laplacian(MatrixAccessor accessor, int x, int y, int z,
    double dx, double dy, double dz) {

    const double doubled = 2 * accessor.get(x, y, z, true);
    return (accessor.get(x-1, y, z, true) - doubled + accessor.get(x+1, y, z, true)) / sqr(dx) +
           (accessor.get(x, y-1, z, true) - doubled + accessor.get(x, y+1, z, true)) / sqr(dy) +
           (accessor.get(x, y, z-1, true) - doubled + accessor.get(x, y, z+1, true)) / sqr(dz);
}

double calculate_laplacian(MatrixAccessor accessor, int x, int y, int z,
    double dx, double dy, double dz, double *array) {

    const int base_index = accessor.derive_index(x, y, z, true);
    const double doubled = 2 * array[base_index];
    return (array[accessor.derive_index(x-1, y, z, true)] - doubled + array[accessor.derive_index(x+1, y, z, true)]) / sqr(dx) +
           (array[accessor.derive_index(x, y-1, z, true)] - doubled + array[accessor.derive_index(x, y+1, z, true)]) / sqr(dy) +
           (array[base_index - 1] - doubled + array[base_index + 1]) / sqr(dz);
}

// direction is represented by 0, 1, 2 => supposed to receive integers in that range
inline AdjacentDirections get_adjacent_directions(int pivot_direction) {

    AdjacentDirections result;
    if (pivot_direction == 1) {
        result.lowest  = 0;
        result.highest = 2;
    } else if (pivot_direction == 2) {
        result.lowest  = 0;
        result.highest = 1;
    } else if (not pivot_direction) {
        result.lowest  = 1;
        result.highest = 2;
    }

    return result;
}

int get_digit_count(int source) {

    int reduced = source;
    int digit_count = 0;
    while (reduced) {
        reduced /= 10;
        digit_count++;
    }

    return (not digit_count)? 1: digit_count;
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

    omp_set_num_threads(param_nt);

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

    // setting offsets for convenient loop indexing
    prev_accessor.configure_offsets(-pb.extended.x.from, -pb.extended.y.from, -pb.extended.z.from);
    curr_accessor.configure_offsets(-pb.extended.x.from, -pb.extended.y.from, -pb.extended.z.from);
    next_accessor.configure_offsets(-pb.extended.x.from, -pb.extended.y.from, -pb.extended.z.from);

    // every block in non-trivial case has 6 neighbors: left-right, front-back, top-bottom
    const int interface_size = 6 * sqr(std::max(
        block_cells[0], std::max(block_cells[1], block_cells[2])));

    double *egress_buffer = new double[interface_size],
          *ingress_buffer = new double[interface_size];

    // zero step: initials and boundaries
    #pragma omp parallel for
    for (int i = pb.extended.x.from; i < pb.extended.x.to; ++i) {
        for (int j = pb.extended.y.from; j < pb.extended.y.to; ++j) {
            for (int k = pb.extended.z.from; k < pb.extended.z.to; ++k) {

                double backfill = u_analytical(i * dx, j * dy, k * dz, 0);
                bool is_frontier_node =
                    i == 0 or j == 0 or k == 0 or
                    i == param_x_nodes or j == param_y_nodes or k == param_z_nodes;

                if (not is_frontier_node) {
                    prev_accessor.set(i, j, k, backfill, true);

                } else {
                    prev_accessor.set(i, j, k, backfill, true);
                    curr_accessor.set(i, j, k, backfill, true);
                    next_accessor.set(i, j, k, backfill, true);
                }
            }
        }
    }

    // first step: laplacian for phi function
    #pragma omp parallel for
    for (int i = pb.inner.x.from; i < pb.inner.x.to; ++i) {
        for (int j = pb.inner.y.from; j < pb.inner.y.to; ++j) {
            for (int k = pb.inner.z.from; k < pb.inner.z.to; ++k) {
                curr_accessor.set(i, j, k,
                    prev_accessor.get(i, j, k, true) + sqr(dt) * calculate_laplacian(
                        prev_accessor, i, j, k, dx, dy, dz) / 2, true);
            }
        }
    }

    // main loop
    double *deltas = new double[param_t_steps + 1];
    double start_time = MPI_Wtime();
    int step = 1;

    while (true) {

        MPI_Request requests[6 * 2];
        int cells_packed = 0,
            cells_claimed = 0;
        int total_requests = 0;
        int rank_src, rank_dst;

        for (int dir = 0; dir < 3; ++dir) {

            const AdjacentDirections ad = get_adjacent_directions(dir);
            const int dir1 = ad.lowest, dir2 = ad.highest;
            // iterating over allowed dispositions
            for (int disp = DISP_DOWNWARDS; disp < DISP_UPWARDS + 1; disp += 2) {

                MPI_Cart_shift(grid_comm, dir, disp, &rank_src, &rank_dst);
                if (rank_dst != MPI_PROC_NULL) {

                    // proactive claim for data transmission
                    int ingress_batch_size = (pb.inner[dir1].to - pb.inner[dir1].from) *
                        (pb.inner[dir2].to - pb.inner[dir2].from);
                    MPI_Irecv(&ingress_buffer[cells_claimed], ingress_batch_size,
                        MPI_DOUBLE, rank_dst, MPI_ANY_TAG, grid_comm, &requests[total_requests++]);
                    cells_claimed += ingress_batch_size;

                    double *batch_start = &egress_buffer[cells_packed];
                    int batch_cells_packed = 0;

                    // iidx == interface index
                    int iidx[3] = {0, 0, 0};
                    if (disp == DISP_UPWARDS) {
                        iidx[dir] = pb.inner[dir].to - 1;
                    } else {
                        iidx[dir] = pb.inner[dir].from;
                    }

                    for (int idx1 = pb.inner[dir1].from; idx1 < pb.inner[dir1].to; ++idx1) {
                        iidx[dir1] = idx1;
                        for (int idx2 = pb.inner[dir2].from; idx2 < pb.inner[dir2].to; ++idx2) {
                            iidx[dir2] = idx2;

                            egress_buffer[cells_packed + batch_cells_packed] =
                                u_calc_curr[curr_accessor.derive_index(
                                    iidx[0], iidx[1], iidx[2], true)];

                            batch_cells_packed++;
                        }
                    }

                    MPI_Isend(batch_start, batch_cells_packed, MPI_DOUBLE, rank_dst, 0,
                        grid_comm, &requests[total_requests]);
                    cells_packed += batch_cells_packed;
                    total_requests++;
                }
            }
        }

        // all the communications must be completed before processing arrived cells;
        // ignorance is to avoid the send/receive details examination
        if (total_requests) {
            MPI_Waitall(total_requests, requests, MPI_STATUSES_IGNORE);
        }

        int cells_unpacked = 0;
        for (int dir = 0; dir < 3; ++dir) {

            const AdjacentDirections ad = get_adjacent_directions(dir);
            const int dir1 = ad.lowest, dir2 = ad.highest;
            for (int disp = DISP_DOWNWARDS; disp < DISP_UPWARDS + 1; disp += 2) {

                MPI_Cart_shift(grid_comm, dir, disp, &rank_src, &rank_dst);
                if (rank_dst != MPI_PROC_NULL) {

                    int iidx[3] = {0, 0, 0};
                    if (disp == DISP_UPWARDS) {
                        iidx[dir] = pb.extended[dir].to - 1;
                    } else {
                        iidx[dir] = pb.extended[dir].from;
                    }

                    for (int idx1 = pb.inner[dir1].from; idx1 < pb.inner[dir1].to; ++idx1) {
                        iidx[dir1] = idx1;
                        for (int idx2 = pb.inner[dir2].from; idx2 < pb.inner[dir2].to; ++idx2) {
                            iidx[dir2] = idx2;

                            u_calc_curr[(iidx[0] - pb.extended.x.from) * block_cells[1] * block_cells[2] +
                                        (iidx[1] - pb.extended.y.from) * block_cells[2] +
                                        (iidx[2] - pb.extended.z.from)] =
                                ingress_buffer[cells_unpacked];
                            cells_unpacked++;
                        }
                    }
                }
            }
        }

        #pragma omp parallel for
        for (int i = pb.inner.x.from; i < pb.inner.x.to; ++i) {
            for (int j = pb.inner.y.from; j < pb.inner.y.to; ++j) {
                for (int k = pb.inner.z.from; k < pb.inner.z.to; ++k) {

                    u_calc_next[next_accessor.derive_index(i, j, k, true)] =
                        calculate_laplacian(curr_accessor, i, j, k, dx, dy, dz, u_calc_curr) * sqr(dt) -
                            u_calc_prev[prev_accessor.derive_index(i, j, k, true)] +
                            2 * u_calc_curr[curr_accessor.derive_index(i, j, k, true)];
                }
            }
        }

        double local_delta = 0;
        #pragma omp parallel for shared(local_delta)
        for (int i = pb.inner.x.from; i < pb.inner.x.to; ++i) {
            for (int j = pb.inner.y.from; j < pb.inner.y.to; ++j) {
                for (int k = pb.inner.z.from; k < pb.inner.z.to; ++k) {

                    double delta = fabs(
                        u_analytical(i * dx, j * dy, k * dz, step * dt) -
                        u_calc_curr[(i - pb.extended.x.from) * block_cells[1] * block_cells[2] +
                            (j - pb.extended.y.from) * block_cells[2] + (k - pb.extended.z.from)]);

                    #pragma omp critical
                    local_delta = (delta > local_delta)? delta: local_delta;
                }
            }
        }

        // getting overall area error
        double global_delta;
        if (nproc == 1) {
            global_delta = local_delta;
        } else {
            MPI_Reduce(&local_delta, &global_delta, 1, MPI_DOUBLE, MPI_MAX, 0, grid_comm);
        }

        deltas[step] = global_delta;

        if (++step > param_t_steps) {
            break;
        }

        double *tmp = u_calc_prev;
        // shifting the arrays
        u_calc_prev = u_calc_curr;
        u_calc_curr = u_calc_next;
        u_calc_next = tmp;
    }

    if (rank == MASTER_PROCESS) {
        for (int i = 1; i <= param_t_steps; ++i) {
            std::cout << std::setw(get_digit_count(param_t_steps)) << i << ": " <<
                std::setprecision(16) << std::fixed << deltas[i] << std::endl;
        }
        std::cout << "Elapsed time: " << std::setprecision(8) << 
            (MPI_Wtime() - start_time) << " s" << std::endl;
    }

    delete [] u_calc_prev;
    delete [] u_calc_curr;
    delete [] u_calc_next;

    delete [] egress_buffer;
    delete [] ingress_buffer;

    delete [] deltas;

    MPI_Finalize();
    exit(SUCCESS);
}
