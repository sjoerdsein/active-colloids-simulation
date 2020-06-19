#include <random>
#include <iostream>
#include <iomanip>
#include <boost/python/numpy.hpp>
#include <boost/multi_array.hpp>
#include "voro++_2d.hh"

// Abbreviate Python and NumPy namespaces
namespace py = boost::python;
namespace np = py::numpy;


// Define some exceptions for easy error handling
class  type_error : public std::runtime_error { using runtime_error::runtime_error; };
class value_error : public std::runtime_error { using runtime_error::runtime_error; };
class shape_error : public type_error { using type_error::type_error; };

void translator( type_error const & e) { PyErr_SetString(PyExc_TypeError , e.what()); }
void translator(value_error const & e) { PyErr_SetString(PyExc_ValueError, e.what()); }


// Define global constants
constexpr double pi = 3.141592653589793238462643383279502884197169399375105820974;
constexpr double two_to_the_one_third = 1.259921049894873164767210607278228350570;
constexpr double d = 1.0; // diameter
constexpr double e = 1.0; // ~ repulsion strength


// Constexpr recursive function for integer powers
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
[[nodiscard]] constexpr vec   operator+ (vec    const lhs,  vec    const rhs) noexcept { return {lhs.x + rhs.x, lhs.y + rhs.y}; }
[[nodiscard]] constexpr vec   operator- (vec    const lhs,  vec    const rhs) noexcept { return {lhs.x - rhs.x, lhs.y - rhs.y}; }
[[nodiscard]] constexpr vec   operator* (vec    const lhs,  double const rhs) noexcept { return {lhs.x * rhs  , lhs.y * rhs  }; }
[[nodiscard]] constexpr vec   operator* (double const lhs,  vec    const rhs) noexcept { return rhs * lhs; }
              constexpr vec & operator+=(vec        & lhs,  vec    const rhs) noexcept { return lhs = lhs + rhs; }
              constexpr vec & operator-=(vec        & lhs,  vec    const rhs) noexcept { return lhs = lhs - rhs; }
              constexpr vec & operator*=(vec        & lhs,  double const rhs) noexcept { return lhs = lhs * rhs; }
[[nodiscard]] constexpr vec   operator- (vec const v) noexcept { return {-v.x, -v.y}; }
[[nodiscard]] constexpr double norm_sq(vec const p) noexcept { return p.x*p.x + p.y*p.y; }

struct particle {
	vec p{};
	double a{};
};

// To store both the pointer to the particle as well as it's position in the
// particles vector
struct particle_ptr_id { particle * ptr; size_t id; };

// Define global variables
static vec border{};
static std::mt19937_64 g{}; // { std::random_device{}() };

// Calculate the index of the bin a particle should go into based on the
// particle's position. WARNING: this will be incorrect if the particle is
// outside of the boundaries
int pos2bin(particle p, int bin_size)
{
	return int(p.p.x/border.x*bin_size) + int(p.p.y/border.y*bin_size)*bin_size;
}

// The WCA force for particles at a certain distance vector
[[nodiscard]] constexpr vec WCA_force(vec dist, double e) noexcept
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
[[nodiscard]] vec interaction_force(std::vector<std::vector<particle_ptr_id>> const & bins,
									std::vector<int> const & bin_idx,
									std::vector<int> const & part_idx,
									int bin_size,
									size_t particle_idx) noexcept
{
	vec total_force = {0.0,0.0};
	// Sum WCA_force for all particles in own and surrounding bins incl PBC, but not itself
	int const bin_index = bin_idx[particle_idx];
	particle const & p = *bins[bin_index][part_idx[particle_idx]].ptr;
	struct v {
		int x, y;
		[[nodiscard]] constexpr v operator+(v const r) const noexcept { return {x + r.x, y + r.y}; }
  };
	v const bin_coord{bin_index % bin_size,
	                  bin_index / bin_size};
	// Loop over surrounding bins
	for (auto const neighbor_offset : {
	     v{-1,-1}, v{0,-1}, v{1,-1},
	     v{-1, 0}, v{0, 0}, v{1, 0},
	     v{-1, 1}, v{0, 1}, v{1, 1}})
	{
		// Periodic boundary conditions for the bins
		vec pbc_correction{0.0,0.0};
		auto bin_pbc = bin_coord + neighbor_offset;
		if (bin_pbc.x <  bin_size) { bin_pbc.x += bin_size; pbc_correction.x -= border.x; }
		if (bin_pbc.x >= bin_size) { bin_pbc.x -= bin_size; pbc_correction.x += border.x; }
		if (bin_pbc.y <  bin_size) { bin_pbc.y += bin_size; pbc_correction.y -= border.y; }
		if (bin_pbc.y >= bin_size) { bin_pbc.y -= bin_size; pbc_correction.y += border.y; }
		// for all particles in the bin, if it isn't p, add the wca force
		for (particle_ptr_id ppi : bins[bin_pbc.x + bin_pbc.y*bin_size]) {
			if (ppi.ptr != &p) total_force += WCA_force(p.p - ppi.ptr->p - pbc_correction);
		}
	}

	return total_force;
}

void simulation_round(std::vector<particle> & particles,
					  std::vector<std::vector<particle_ptr_id>> & bins,
                      std::vector<int> & bin_idx,
                      std::vector<int> & part_idx,
                      int bin_size,
                      std::normal_distribution<double> & dx_prng,
                      std::normal_distribution<double> & da_prng,
                      double interaction_step_scale,
                      double propulsion_strength)
{
	for (size_t i{}; i < particles.size(); ++i) {
		// abbreviation
		particle & p { particles[i] };

		// apply random rotation
		auto const da_rand = da_prng(g);
		p.a += da_rand;

		// calculate the offsets from the forces
		auto const dx_rand = vec{dx_prng(g), dx_prng(g)};
		auto const dx_int  = interaction_step_scale * interaction_force(bins, bin_idx, part_idx, bin_size, i);
		auto const dx_prop = propulsion_strength * vec{std::cos(p.a), std::sin(p.a)};

		// apply offsets
		vec & v = p.p;
        //std::cout << norm_sq(dx_rand) << ' ' << norm_sq(dx_int) << ' ' << norm_sq(dx_prop) << '\n';
		v += dx_rand + dx_int + dx_prop;

		// apply periodic boundary conditions
		// exit when we are about to get a near-infinite loop
		for (int i{}; v.x <  border.x; ++i) { v.x += border.x; if (i>2) throw std::runtime_error{"particle is too far left"}; }
		for (int i{}; v.x >= border.x; ++i) { v.x -= border.x; if (i>2) throw std::runtime_error{"particle is too far right"}; }
		for (int i{}; v.y <  border.y; ++i) { v.y += border.y; if (i>2) throw std::runtime_error{"particle is too far down"}; }
		for (int i{}; v.y >= border.y; ++i) { v.y -= border.y; if (i>2) throw std::runtime_error{"particle is too far up"}; }

		if (int const old_idx=bin_idx[i], new_idx=pos2bin(p, bin_size); new_idx != old_idx){
			bins[new_idx].emplace_back(std::move(bins[old_idx][part_idx[i]]));
			auto next_part = bins[old_idx].erase(bins[old_idx].begin() + part_idx[i]);
			for (; next_part != bins[old_idx].end(); ++next_part) {
				part_idx[next_part->id] = next_part - bins[old_idx].begin();
			}
			bin_idx[i] = new_idx;
			part_idx[i] = bins[new_idx].size()-1;
		}
	}
}

void save_frame(std::vector<particle> & particles,
                boost::multi_array_ref<double, 3> & archive_data,
                voro::container_2d & con,
                int frame_idx)
{
	// Save this frame's positions to return to python
	// Reinterpret cast is probably UB
	boost::const_multi_array_ref<double, 2> const particles_data(reinterpret_cast<double const *>(particles.data()),
                                                               boost::extents[particles.size()][3]);
	archive_data[frame_idx] = particles_data;

	// Calculate density using voronoi diagram

	// Add particles to voronoi
	for (int i{}; auto const & p : particles) {
		con.put(i++, p.p.x, p.p.y);
	}
	// For all particles, calculate and save the cell's area and density
	voro::c_loop_all_2d vl(con);
	voro::voronoicell_2d c{};
	if(vl.start()) do if(con.compute_cell(c,vl)) {
		archive_data[frame_idx][con.id[vl.ij][vl.q]][2] = 1.0/c.area();
	} while(vl.inc());

	// Clear the diagram for the next round
	con.clear();
}

// The main simulation function that is called from Python
auto simulate(py::list const box_size,
              np::ndarray const init,
              double viscosity,
              double propulsion_strength,
              double Dt,
              double density_scale_factor,
              int nr_densities,
              int frames_per_density,
              int frame_interval,
              int init_equil_rounds,
              int density_equil_rounds)
{
	// Verify the types, dimensions and values of the arguments
	if (init.get_dtype()  != np::dtype::get_builtin<double>())
	                            { throw  type_error{"array dtype must be double"    }; }
	if (init.get_nd()     != 2) { throw shape_error{"array must be two-dimensional" }; }
	if (init.shape(1)     != 3) { throw shape_error{"array must be of shape (3, N)" }; }
	if (init.shape(0)     <= 0) { throw value_error{"array must not be empty"       }; }
	if (py::len(box_size) != 2) { throw shape_error{"box size list must be length 2"}; }

	if (viscosity <= 0.) { throw value_error{"viscosity must have a positive non-zero value"}; }

	border = {py::extract<double>(box_size[0]),
	          py::extract<double>(box_size[1])};

	if (std::isnan(border.x) or border.x <= 0 or std::isnan(border.y) or border.y <= 0) {
		throw value_error{"invalid box size: either NaN or <= 0"};
	}

	// Copy the data into a std::vector
	std::cout << "Copying data from Python to C++\n";
	auto init_data { reinterpret_cast<particle const *>(init.get_data()) }; // UB?
	std::vector<particle> particles(init_data, init_data + init.shape(0)); // copy

	// Create a Boost.MultiArray to store snapshots of the simulation
	int const nr_frames = nr_densities * frames_per_density;
	np::ndarray archive { np::empty(py::make_tuple(nr_frames, init.shape(0), 3), init.get_dtype()) };
	boost::multi_array_ref<double, 3> archive_data(reinterpret_cast<double *>(archive.get_data()),
	                                               boost::extents[nr_frames][init.shape(0)][3]);

	std::cout << "Generating bins...\n";

	// Fill the bins with pointers to their particles
	// We can probably exploit voro++ to do this for us, but no
	int const bin_size = std::round(std::cbrt(particles.size()));
	std::vector<std::vector<particle_ptr_id>> bins(bin_size*bin_size);
	std::vector<int> bin_idx (particles.size());
	std::vector<int> part_idx(particles.size());
	for (auto & bin : bins) bin.reserve(2*bin_size);
	for (size_t i{}; i < particles.size(); ++i) {
		particle & p { particles[i] };
		int const bi = pos2bin(p, bin_size);
		bin_idx[i] = bi; // In what bin?
		try { bins.at(bi).emplace_back(particle_ptr_id{&p, i}); } // Store
		catch (std::out_of_range const &) {
            std::cerr << "It appears a particle is not inside the boundaries\n";
            throw;
		}
		part_idx[i] = bins[bi].size()-1; // Where in the bin?
	} //*/

	std::cout << "Generated " << bin_size << " x " << bin_size << " bins\n";

	// Prepare for the numerical integration
	double const gamma = 3.0 * pi * d * viscosity;  // Friction coefficient
	double const random_step_scale = std::sqrt(Dt / gamma * 2.0); // Translational diffusion coefficient
	double const interaction_step_scale = Dt / gamma;
	double const rotation_step_scale = std::sqrt(Dt / (pi * d*d*d * viscosity) * 2.0); // Rotational diffusion coefficient

	std::cout << "Random step scale    : " << random_step_scale << "\nPropulsion step scale: " << propulsion_strength << '\n';

	std::normal_distribution<double> dx_prng(0.0, random_step_scale);
	std::normal_distribution<double> da_prng(0.0, rotation_step_scale);

	// Add voronoi container for density calculations
	voro::container_2d con(0, border.x, 0, border.y, 8, 8, true, true, init.shape(0)*2/8/8);
	//                     {container borders x, y} {# div} {periodic} {# particles per div}

	// Abbreviate this function call to `sim(repeat)`
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


// Define our Python module
BOOST_PYTHON_MODULE(mcexercise)
{
	np::initialize();
	py::def("simulate", simulate);

	py::register_exception_translator< type_error>(static_cast<void (*)( type_error const &)>(translator));
	py::register_exception_translator<value_error>(static_cast<void (*)(value_error const &)>(translator));
}

