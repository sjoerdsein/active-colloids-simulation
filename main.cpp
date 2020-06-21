/*
This file contains all code that must be fast: the simulation of the particles
and the calculation of the Voronoi diagram. This file is most easily understood
by first reading to the pos2bin() function, and then starting at the bottom and
reading through every function moving up the file.

We first get our data from Python through the Boost.Python libraries. Python
effectively calls our simulate() function with all parameters we need to know
to perform the simulation. This includes the initial positions and orientations
of the particles, in the form of a NumPy array.

To improve the speed of the simulation we try to waste as little resources as
possible. E.g. when calculating the interaction force on a particle, we do not
check for every single particle if they are close enough to exert any force (O(n^2)),
but we only check those particles which actually have a good chance of being
close enough (O(n)). This is done by sorting pointers to the particles in a two
dimensional grid, and only checking those particles which are in the grid cells
near the particle whose force we want to know. The particles are sorted into
the grid only once, and updated if they are moved.
*/

// Include STL headers
#include <random>
#include <iostream>
#include <iomanip>
// Include Boost.Python and Boost.MultiArray
#include <boost/python/numpy.hpp>
#include <boost/multi_array.hpp>
// Include Voro++
#include "voro++_2d.hh"

// Abbreviate the Python and NumPy namespaces
namespace py = boost::python;
namespace np = py::numpy;


// Define some exceptions for easy error handling
class  type_error : public std::runtime_error { using runtime_error::runtime_error; };
class value_error : public std::runtime_error { using runtime_error::runtime_error; };
class shape_error : public type_error { using type_error::type_error; };
// And some translator functions for Boost.Python
void translator( type_error const & e) { PyErr_SetString(PyExc_TypeError , e.what()); }
void translator(value_error const & e) { PyErr_SetString(PyExc_ValueError, e.what()); }


// Define global constants
constexpr double pi = 3.141592653589793238462643383279502884197169399375105820974;
constexpr double two_to_the_one_third = 1.259921049894873164767210607278228350570;
constexpr double d = 1.0; // diameter
constexpr double e = 1.0; // ~ repulsion strength


// Constexpr recursive function for integer powers (home made)
constexpr bool fast_power = true; // Slightly less precise, but reduces fifteen multiplications to four
template <int p>
[[nodiscard]] constexpr auto to_the(auto const base) noexcept requires(p >= 0) {
	if constexpr (p == 0) return 1.0;
	else if constexpr (fast_power and p % 2 == 0) return to_the<p/2>(base * base);
	else return to_the<p-1>(base) * base;
}

// Define a mathematical 2D vector object and its operations
struct vec {
	double x{}, y{};
};
[[nodiscard]] constexpr vec   operator+ (vec    const lhs,  vec    const rhs) noexcept { return {lhs.x + rhs.x, lhs.y + rhs.y}; } // Vector addition
[[nodiscard]] constexpr vec   operator- (vec    const lhs,  vec    const rhs) noexcept { return {lhs.x - rhs.x, lhs.y - rhs.y}; } // Vector subtraction
[[nodiscard]] constexpr vec   operator* (vec    const lhs,  double const rhs) noexcept { return {lhs.x * rhs  , lhs.y * rhs  }; } // Scalar-vector multiplication
[[nodiscard]] constexpr vec   operator* (double const lhs,  vec    const rhs) noexcept { return rhs * lhs; }                      // Vector-scalar multiplication
              constexpr vec & operator+=(vec        & lhs,  vec    const rhs) noexcept { return lhs = lhs + rhs; }                // Vector addition assignment
              constexpr vec & operator-=(vec        & lhs,  vec    const rhs) noexcept { return lhs = lhs - rhs; }                // Vector subtraction assignment
              constexpr vec & operator*=(vec        & lhs,  double const rhs) noexcept { return lhs = lhs * rhs; }                // Vector-scalar multiplication assignment
[[nodiscard]] constexpr vec   operator- (vec const v) noexcept { return {-v.x, -v.y}; }                                           // Vector negation
[[nodiscard]] constexpr double norm_sq(vec const p) noexcept { return p.x*p.x + p.y*p.y; }                                        // Norm squared

// A particle object has a position vector and an angle
struct particle {
	vec p{};
	double a{};
};

// Define global variables
static vec border{}; // The lengths of the box. The box is assumed to start as (0,0)
static std::mt19937_64 g{}; // The random bit generator, used to later generate random numbers

// NOW READ FROM THE BOTTOM UP
// and follow the functions as they are called
// --------------------------------------------
// THE END

// Calculate the index of the bin a particle should go into based on the
// particle's position. WARNING: this will be incorrect if the particle is
// outside of the boundaries
int pos2bin(particle p, int bin_size)
{
	return int(p.p.x/border.x*bin_size) + int(p.p.y/border.y*bin_size)*bin_size;
}

// The WCA force for particles at a certain distance vector
[[nodiscard]] constexpr vec WCA_force(vec dist) noexcept
{
	double const r2 = norm_sq(dist);

	if (r2 >= two_to_the_one_third * d*d) return {0,0};

	double const r6  = to_the<3>(r2);
	double const r14 = to_the<7>(r2);
	constexpr double d6 = to_the<6>(d);

	return dist * (24.0*e*d6*(d6+d6-r6)/r14); // ~23% faster
	//return dist * (4*e*(12*d12/r14 - 6*d6/r8)); // equivalent
}

// The total WCA interaction force between a particle and all its neighbours,
// accounting for periodic boundary conditions
[[nodiscard]] vec interaction_force(std::vector<std::vector<particle *>> const & bins,
                                    std::vector<int> const & bin_idx,
                                    std::vector<int> const & part_idx,
                                    int bin_size,
                                    size_t particle_idx) noexcept
{
	// Define a quick integer vector object, for some cleaner code later on
	struct v {
		int x, y;
		[[nodiscard]] constexpr v operator+(v const r) const noexcept { return {x + r.x, y + r.y}; }
  };

	vec total_force = {0.0,0.0}; // The old type of vector to accumulate the forces
	// Sum WCA_force for all particles in own and surrounding bins incl PBC, but not itself
	int const bin_index = bin_idx[particle_idx]; // In what bin is this particle stored?
	particle const * p = bins[bin_index][part_idx[particle_idx]]; // Get a reference to the particle we are exerting the force on
	v const bin_coord{bin_index % bin_size, // What would be the 2D grid-coordinates of this bin
	                  bin_index / bin_size};
	// Loop over surrounding bins
	for (auto const neighbor_offset : {
	     v{-1,-1}, v{0,-1}, v{1,-1},
	     v{-1, 0}, v{0, 0}, v{1, 0},
	     v{-1, 1}, v{0, 1}, v{1, 1}})
	{
		// Periodic boundary conditions for the whole bin
		vec pbc_correction{0.0,0.0};
		auto bin_pbc = bin_coord + neighbor_offset;
		if (bin_pbc.x <  bin_size) { bin_pbc.x += bin_size; pbc_correction.x -= border.x; }
		if (bin_pbc.x >= bin_size) { bin_pbc.x -= bin_size; pbc_correction.x += border.x; }
		if (bin_pbc.y <  bin_size) { bin_pbc.y += bin_size; pbc_correction.y -= border.y; }
		if (bin_pbc.y >= bin_size) { bin_pbc.y -= bin_size; pbc_correction.y += border.y; }
		// For all particles in the bin, if it isn't p, add the WCA force
		// p is the particle the forces apply to; pp is a particle in one of the surrounding bins
		for (particle * pp : bins[bin_pbc.x + bin_pbc.y*bin_size]) {
			if (pp != p) total_force += WCA_force(p->p - pp->p - pbc_correction);
		}
	}

	return total_force;
}

// This function performs the actual simulation for a single frame: looping
// over all particles and applying their forces and offsets
void simulation_round(std::vector<particle> & particles,
                      std::vector<std::vector<particle *>> & bins,
                      std::vector<int> & bin_idx,
                      std::vector<int> & part_idx,
                      int bin_size,
                      std::normal_distribution<double> & dx_prng,
                      std::normal_distribution<double> & da_prng,
                      double interaction_step_scale,
                      double propulsion_strength)
{
	for (size_t i{}; i < particles.size(); ++i) {
		// Abbreviation
		particle & p { particles[i] };

		// Get and apply random rotation
		auto const da_rand = da_prng(g);
		p.a += da_rand;

		// Calculate the offsets from the forces
		auto const dx_rand = vec{dx_prng(g), dx_prng(g)}; // The random force
		auto const dx_int  = interaction_step_scale * interaction_force(bins, bin_idx, part_idx, bin_size, i); // The interaction force
		auto const dx_prop = propulsion_strength * vec{std::cos(p.a), std::sin(p.a)}; // The propulsion force

		// Apply the offsets
		vec & v = p.p;
		v += dx_rand + dx_int + dx_prop;

		// Apply periodic boundary conditions
		// Exit when we are about to get a near-infinite loop
		for (int i{}; v.x <  border.x; ++i) { v.x += border.x; if (i>10) throw std::runtime_error{"particle is too far left"}; }
		for (int i{}; v.x >= border.x; ++i) { v.x -= border.x; if (i>10) throw std::runtime_error{"particle is too far right"}; }
		for (int i{}; v.y <  border.y; ++i) { v.y += border.y; if (i>10) throw std::runtime_error{"particle is too far down"}; }
		for (int i{}; v.y >= border.y; ++i) { v.y -= border.y; if (i>10) throw std::runtime_error{"particle is too far up"}; }

		// Reorder the particle in the bins, if necessary
		if (int const old_idx=bin_idx[i], new_idx=pos2bin(p, bin_size); new_idx != old_idx){ // Find the correct position of the particle, and change it if needed
			bins[new_idx].emplace_back(std::move(bins[old_idx][part_idx[i]])); // Add the particle to the back of its new bin
			auto next_part = bins[old_idx].erase(bins[old_idx].begin() + part_idx[i]); // Remove the particle from the old bin
			for (; next_part != bins[old_idx].end(); ++next_part) {
				part_idx[*next_part - particles.data()] = next_part - bins[old_idx].begin(); // All following particles in this bin just shifted one place forward, so change the part_idx array to reflect that
			}
			bin_idx[i] = new_idx; // Save to what bin the particle was moved
			part_idx[i] = bins[new_idx].size()-1; // Save to where in the bin the particle was moved
		}
	}
}

// This function saves the positions of all particles to the archive NumPy
// array and also calculates and adds their densities
void save_frame(std::vector<particle> & particles,
                boost::multi_array_ref<double, 3> & archive_data,
                voro::container_2d & con,
                int frame_idx)
{
	// Save this frame's positions to return to python
	// Reinterpret cast is probably UB
	boost::const_multi_array_ref<double, 2> const particles_data(reinterpret_cast<double const *>(particles.data()),
                                                               boost::extents[particles.size()][3]);
	archive_data[frame_idx] = particles_data; // Copy the data

	// Calculate density using voronoi diagram

	// Add particles to voronoi
	for (int i{}; auto const & p : particles) {
		con.put(i++, p.p.x, p.p.y);
	}
	// For all particles, calculate and save the cell's area and density,
	// overwriting the third coordinate which used to be their angles
	voro::c_loop_all_2d vl(con);
	voro::voronoicell_2d c{};
	if(vl.start()) do if(con.compute_cell(c,vl)) {
		archive_data[frame_idx][con.id[vl.ij][vl.q]][2] = 1.0/c.area();
	} while(vl.inc());

	// Clear the diagram for the next round
	con.clear();
}

// The main simulation function that is called from Python
auto simulate(py::list const box_size,     // The size of the boundary
              np::ndarray const init,      // Initial positions and orientations of the particles
              double viscosity,
              double propulsion_strength,
              double Dt,
              double density_scale_factor, // XXX
              int nr_densities,            // XXX
              int frames_per_density,      // FIXME
              int frame_interval,
              int init_equil_rounds,
              int density_equil_rounds)    // XXX
{

	int const nr_particles = init.shape(0);
	// Verify the types, dimensions and values of the arguments
	if (init.get_dtype()  != np::dtype::get_builtin<double>())
	                            { throw  type_error{"array dtype must be double"    }; }
	if (init.get_nd()     != 2) { throw shape_error{"array must be two-dimensional" }; }
	if (init.shape(1)     != 3) { throw shape_error{"array must be of shape (3, N)" }; }
	if (nr_particles      <= 0) { throw value_error{"array must not be empty"       }; }
	if (py::len(box_size) != 2) { throw shape_error{"box size list must be length 2"}; }

	if (viscosity <= 0.) { throw value_error{"viscosity must have a positive non-zero value"}; }
	if (propulsion_strength < 0.) { throw value_error{"propulsion strength must have a positive value"}; }
	if (Dt <= 0.) { throw value_error{"delta time must have a positive non-zero value"}; }
	if (frames_per_density < 0) { throw value_error{"frames per density must have a positive value"}; }
	if (frame_interval <= 0) { throw value_error{"frame interval must have a positive non-zero value"}; }
	if (init_equil_rounds < 0) { throw value_error{"equalization rounds must have a positive value"}; }

	// Extract the size of the border from the Python list and store it in a vec
	border = {py::extract<double>(box_size[0]),
	          py::extract<double>(box_size[1])};
	// Verify the values are valid
	if (std::isnan(border.x) or border.x <= 0 or std::isnan(border.y) or border.y <= 0) {
		throw value_error{"invalid box size: either NaN or <= 0"};
	}

	// Copy the data from init into a std::vector of particles
	std::cout << "Copying data from Python to C++\n";
	auto init_data { reinterpret_cast<particle const *>(init.get_data()) }; // UB? Get the start pointer of the data
	std::vector<particle> particles(init_data, init_data + nr_particles);   // Copy data from start to start+length

	// Create an empty NumPy array to store snapshots of the simulation, and a
	// Boost.MultiArray ref to easily access it's data within C++
	int const nr_frames = nr_densities * frames_per_density;                                          // The number of snapshots to be stored
	np::ndarray archive { np::empty(py::make_tuple(nr_frames, nr_particles, 3), init.get_dtype()) };  // Create the NumPy array at it's final size
	boost::multi_array_ref<double, 3> archive_data(reinterpret_cast<double *>(archive.get_data()),    // Also create Boost.MultiArray to reference the NumPy array
	                                               boost::extents[nr_frames][nr_particles][3]);

	std::cout << "Generating bins...\n";

	// Fill the 2D grid (bins) with pointers to their particles. We can probably
	// exploit voro++ to do this for us, but no
	int const bin_size = std::round(std::cbrt(nr_particles)); // Both the number of bins along each axis, as well as the expected average number of particles in each bin
	std::vector<std::vector<particle *>> bins(bin_size*bin_size); // The actual bins that store the particles' positions in the particles vector
	std::vector<int> bin_idx (nr_particles); // Stores which particle resides in which bin, by the particles index
	std::vector<int> part_idx(nr_particles); // Stores at which index in the bin the particle is stored
	for (auto & bin : bins) bin.reserve(2*bin_size); // Reserve 2x the average expected space in each bin
	for (int i{}; i < nr_particles; ++i) { // Actually filling the bins
		particle & p { particles[i] }; // Abbreviation
		int const bi = pos2bin(p, bin_size); // Find in what bin the particle should be stored
		bin_idx[i] = bi; // Save in what bin the particle will be stored
		try { bins.at(bi).emplace_back(&p); } // Store the particle in the correct bin
		catch (std::out_of_range const &) { // Catch any segmentation faults
            std::cerr << "It appears a particle is not inside the boundaries\n";
            throw;
		}
		part_idx[i] = bins[bi].size()-1; // Store where in the bin the particle was stored
	}

	std::cout << "Generated " << bin_size << " x " << bin_size << " bins\n";

	// Prepare for the numerical integration by calculating some values
	double const gamma = 3.0 * pi * d * viscosity;  // Friction coefficient
	double const random_step_scale = std::sqrt(2.0 * Dt / gamma); // Translational diffusion coefficient
	double const interaction_step_scale = Dt / gamma; // It's just Dt/gamma
	double const rotation_step_scale = std::sqrt(2.0 * Dt / (pi * d*d*d * viscosity)); // Rotational diffusion coefficient

	std::cout << "Random step scale    : " << random_step_scale << "\nPropulsion step scale: " << propulsion_strength << '\n';

	// Instead of multiplying each random value from a unit stddev distribution
	// by a constant, just set this constant as the stddev
	std::normal_distribution<double> dx_prng(0.0, random_step_scale);   // PRNG for the random force
	std::normal_distribution<double> da_prng(0.0, rotation_step_scale); // PRNG for the random torque

	// Add voronoi container for density calculations
	// The points are stored in a grid just like out bins
	voro::container_2d con(0, border.x, 0, border.y, bin_size, bin_size, true, true, 2*bin_size);
	//                     {container borders x, y} {nr of bins per axis}{periodic?} {nr of particles per bin}

	// Abbreviate the call to simulation_round() and add some logging
	auto const sim = [&particles, &bins, &bin_idx, &part_idx, bin_size, &dx_prng, &da_prng, &interaction_step_scale, &propulsion_strength](int const repeat = 1, bool log = false){
		if (repeat < 0) throw std::runtime_error{"repeating a negative number of times"};
		if (log) std::cout << std::fixed << std::setprecision(1);
		for (int i{}; i < repeat; ++i) {
			//if (log) std::cout << "Running inner loop " << i << '/' << repeat << '\n';
			if (log and i % int(repeat/1000.0) == 0) std::cout << "\rRunning inner loop " << i*100.0/repeat << '%' << std::flush;
			simulation_round(particles, bins, bin_idx, part_idx, bin_size, dx_prng, da_prng, interaction_step_scale, propulsion_strength);
		}
		if (log) std::cout << "\rFinished inner loop 100%\n";
	};

	// Initial equilibration
	std::cout << "Equilibrating the system\n";
	sim(init_equil_rounds, /*log*/ true);

	for (int j{}; j < nr_densities; ++j) {
		std::cout << "Running loop " << j << '\n';
		// Equilibration after each density change
		sim(density_equil_rounds);
		// The actual recorded simulation
		for (int i{}; i < frames_per_density * frame_interval; ++i) {
			sim();
			if (i % frame_interval == 0) {
				save_frame(particles, archive_data, con, j*frames_per_density + i/frame_interval);
			}
		}
		// Decrease density
		border *= density_scale_factor;
		std::for_each(particles.begin(), particles.end(),
		              [density_scale_factor](particle & p){ p.p *= density_scale_factor; });
	}

	return archive;
}


// Define our Python module named mcexercise
BOOST_PYTHON_MODULE(mcexercise)
{
	np::initialize();
	py::def("simulate", simulate); // Define the 'simulate function in python and call simulate()

	// Register which exception translator should be called upon which exception
	py::register_exception_translator< type_error>(static_cast<void (*)( type_error const &)>(translator));
	py::register_exception_translator<value_error>(static_cast<void (*)(value_error const &)>(translator));
}

