#include <random>
#include <boost/python/numpy.hpp>
#include <boost/multi_array.hpp>

// Abbreviate Python and NumPy namespaces
namespace py = boost::python;
namespace np = py::numpy;


// Define some exceptions for easy error handling
class  type_error : public std::runtime_error { using runtime_error::runtime_error; };
class value_error : public std::runtime_error { using runtime_error::runtime_error; };
class shape_error : public type_error { using type_error::type_error; };

void translator( type_error const & e) { PyErr_SetString(PyExc_TypeError , e.what()); }
void translator(value_error const & e) { PyErr_SetString(PyExc_ValueError, e.what()); }


// Initialize global variables
constexpr double two_pi = 6.283185307179586476925286766559005768394338798750211641949;
constexpr double two_to_the_one_sixth = 1.1224620483093729814335330496791795162324111;
constexpr double two_to_the_one_third = 1.2599210498948731647672106072782283505702514;
std::mt19937_64 g{}; // std::random_device{}() };
constexpr double d  = 1.0; // diameter
constexpr double e  = 1.0; // ~ repulsion strength


// Constexpr recursive function for integer powers
constexpr bool fast_power = true; // Slightly less precise, but reduces 11 multiplications to four
template <int p>
[[nodiscard]] constexpr auto to_the(auto const base) noexcept requires(p > 0) {
		if constexpr (p == 1) return base;
		else if constexpr (fast_power and p % 2 == 0) return to_the<p/2>(base * base);
		else return to_the<p-1>(base) * base;
}

// Define a 'point' object and its operations
struct point {
	double x{}, y{};
};
[[nodiscard]] constexpr point operator+(point const lhs, point const rhs) noexcept { return {lhs.x + rhs.x, lhs.y + rhs.y}; }
[[nodiscard]] constexpr point operator-(point const lhs, point const rhs) noexcept { return {lhs.x - rhs.x, lhs.y - rhs.y}; }
[[nodiscard]] constexpr point operator*(point const lhs, double const rhs) noexcept { return {lhs.x * rhs, lhs.y * rhs}; }
[[nodiscard]] constexpr point operator*(double const lhs, point const rhs) noexcept { return rhs * lhs; }
constexpr point & operator+=(point & lhs, point const rhs) noexcept { return lhs = lhs + rhs; }
constexpr point & operator-=(point & lhs, point const rhs) noexcept { return lhs = lhs - rhs; }
constexpr point & operator*=(point & lhs, double const rhs) noexcept { return lhs = lhs * rhs; }
[[nodiscard]] constexpr double norm_sq(point const p) noexcept { return p.x*p.x + p.y*p.y; }

// The border as a global point (shouldn't it be called a 'vector')
point border;


// The WCA force for particles at a certain distance vector
[[nodiscard]] constexpr auto WCA_force(point dist) noexcept {
	double const r2 = norm_sq(dist);

	if (r2 >= two_to_the_one_third * d*d) return point{0,0};

	double const r8  = to_the<4>(r2);
	double const r14 = to_the<7>(r2);
	constexpr double const d6  = to_the <6>(d);
	constexpr double const d12 = to_the<12>(d);

	return dist * (4*e*(6*d6/r8 - 12*d12/r14));
}

// The total WCA interaction force between a particle and all its neighbours,
// accounting for periodic boundary conditions
[[nodiscard]] auto interaction_force(std::vector<point> const & points, size_t point_idx) noexcept {
	// Calculate WCA force without boundary conditions, but with a specific offset
	auto calc_force = [&points, point_idx](point const pbc_shift = {0,0}) noexcept {
		point const testpoint { points[point_idx] + pbc_shift };
		point total_force{};
		for (size_t i{}; i < points.size(); ++i) {
			if (i != point_idx) total_force += WCA_force(points[i] - testpoint);
		} return total_force;
	};

	point const p { points[point_idx] };
	constexpr auto force_range = d * two_to_the_one_sixth;

	// Sum WCA forces, and wrap around boundaries if needed
	auto result = calc_force();
	if (p.x < force_range) {
		result += calc_force({ border.x, 0});
		if (p.y < force_range)
			result += calc_force({ border.x,  border.y});
		else if (p.y > border.y-force_range)
			result += calc_force({ border.x, -border.y});
	} else if (p.x > border.x-force_range) {
		result += calc_force({-border.x, 0});
		if (p.y < force_range)
			result += calc_force({-border.x,  border.y});
		else if (p.y > border.y-force_range)
			result += calc_force({-border.x, -border.y});
	}
	if (p.y < force_range) // NO else!
		result += calc_force({0,  border.y});
	else if (p.y > border.y-force_range)
		result += calc_force({0, -border.y});
	return result;
}


// The main simulation function that is called from Python
auto simulate(py::list const box_size, np::ndarray const init, int rounds, int skip_rounds, double gamma, double Dt)
{
	// Verify the types, dimensions and values of the arguments
	if (init.get_dtype()  != np::dtype::get_builtin<double>())
	                            { throw  type_error{"array dtype must be double"    }; }
	if (init.get_nd()     != 2) { throw shape_error{"array must be two-dimensional" }; }
	if (init.shape(1)     != 2) { throw shape_error{"array must be of shape (2, N)" }; }
	if (init.shape(0)     <= 0) { throw value_error{"array must not be empty"       }; }
	if (py::len(box_size) != 2) { throw shape_error{"box size list must be length 2"}; }

	if (gamma <= 0.) { throw value_error{"friction coefficient must have a positive non-zero value"}; }

	border = {py::extract<double>(box_size[0]),
	          py::extract<double>(box_size[1])};

	if (std::isnan(border.x) or border.x <= 0 or std::isnan(border.y) or border.y <= 0) {
		throw value_error{"invalid box size: either NaN or <= 0"};
	}

	// Copy the data into a std::vector
	auto init_data { reinterpret_cast<point const *>(init.get_data()) }; // UB?
	std::vector<point> points(init_data, init_data + init.shape(0)); // copy

	// Create a Boost.MultiArray to store snapshots of the simulation
	int const nr_frames = rounds / skip_rounds;
	np::ndarray archive { np::empty(py::make_tuple(nr_frames, init.shape(0), 2), init.get_dtype()) };
	boost::multi_array_ref<double, 3> archive_data(reinterpret_cast<double *>(archive.get_data()), boost::extents[nr_frames][init.shape(0)][2]);

	// Prepare for the numerical integration
	double const interaction_step_scale = Dt / gamma;
	double const random_step_scale = std::sqrt(Dt / gamma * 2.);

	std::normal_distribution<double> dx_prng(0.0, random_step_scale);

	// Perform the actual simulation
	for (int j{}; j < rounds; ++j) {
		for (size_t i{}; i < points.size(); ++i) {
			// Calculate the offsets from the forces
			auto const dx_rand = point{dx_prng(g), dx_prng(g)};
			auto const dx_int  = interaction_step_scale * interaction_force(points, i);

			// Apply offsets
			point & p { points[i] };
			p += dx_rand + dx_int;

			// Apply periodic boundary conditions
			// Rather slow when the particles are near 10^50...
			while (p.x <  border.x) p.x += border.x;
			while (p.x >= border.x) p.x -= border.x;
			while (p.y <  border.y) p.y += border.y;
			while (p.y >= border.y) p.y -= border.y;
		}
		// Save a snapshot
		if (j % skip_rounds == 0) {
			// Reinterpret cast is probably UB
			boost::const_multi_array_ref<double, 2> const points_data(reinterpret_cast<double const *>(points.data()), boost::extents[points.size()][2]);
			archive_data[j/skip_rounds] = points_data;
		}
	}
	return archive;
}


// Define out Python module
BOOST_PYTHON_MODULE(mcexercise)
{
	np::initialize();
	py::def("simulate", simulate);

	py::register_exception_translator< type_error>(static_cast<void (*)( type_error const &)>(translator));
	py::register_exception_translator<value_error>(static_cast<void (*)(value_error const &)>(translator));
}

