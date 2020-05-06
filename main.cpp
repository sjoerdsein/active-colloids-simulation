#include <random>
#include <boost/python/numpy.hpp>

namespace py = boost::python;
namespace np = py::numpy;


class  type_error : public std::runtime_error { using runtime_error::runtime_error; };
class value_error : public std::runtime_error { using runtime_error::runtime_error; };
class shape_error : public type_error { using type_error::type_error; };

void translator( type_error const & e) { PyErr_SetString(PyExc_TypeError , e.what()); }
void translator(value_error const & e) { PyErr_SetString(PyExc_ValueError, e.what()); }


constexpr double two_pi{ 6.283185307179586476925286766559005768394338798750211641949 };
std::mt19937_64 g { std::random_device{}() };
constexpr double d { 1.0 }; // diameter

struct point {
	double x, y;
	//constexpr auto operator<=>(point const &) const noexcept = default;
};
[[nodiscard]] constexpr point operator+(point const lhs, point const rhs) noexcept { return {lhs.x + rhs.x, lhs.y + rhs.y}; }
[[nodiscard]] constexpr point operator-(point const lhs, point const rhs) noexcept { return {lhs.x - rhs.x, lhs.y - rhs.y}; }
[[nodiscard]] constexpr point operator*(point const lhs, double const rhs) noexcept { return {lhs.x * rhs, lhs.y * rhs}; }
[[nodiscard]] constexpr point operator*(double const lhs, point const rhs) noexcept { return rhs * lhs; }
constexpr point & operator+=(point & lhs, point const rhs) noexcept { return lhs = lhs + rhs; }
constexpr point & operator-=(point & lhs, point const rhs) noexcept { return lhs = lhs - rhs; }
constexpr point & operator*=(point & lhs, double const rhs) noexcept { return lhs = lhs * rhs; }
[[nodiscard]] constexpr double norm_sq(point const p) noexcept { return p.x*p.x + p.y*p.y; }

point border;

[[nodiscard]] bool intersects(size_t point_idx, std::vector<point> const & points, size_t start=0) noexcept
{
	for (size_t i{start}; i < points.size(); ++i) {
		if (i != point_idx and norm_sq(points[i] - points[point_idx]) < d*d) return true;
	}
	return false;
}

auto simulate(py::list const box_size, np::ndarray const init, int rounds)
{
	if (init.get_dtype()  != np::dtype::get_builtin<double>())
	                            { throw  type_error{"array dtype must be double"    }; }
	if (init.get_nd()     != 2) { throw shape_error{"array must be two-dimensional" }; }
	if (init.shape(1)     != 2) { throw shape_error{"array must be of shape (2, N)" }; }
	if (init.shape(0)     <= 0) { throw value_error{"array must not be empty"       }; }
	if (py::len(box_size) != 2) { throw shape_error{"box size list must be length 2"}; }

	border = {py::extract<double>(box_size[0]),
	          py::extract<double>(box_size[1])};

	if (std::isnan(border.x) or border.x <= 0 or std::isnan(border.y) or border.y <= 0) {
		throw value_error{"invalid box size: either NaN or <= 0"};
	}

	auto init_data { reinterpret_cast<point const *>(init.get_data()) }; // UB?
	std::vector<point> points(init_data, init_data + init.shape(0)); // copy

	std::uniform_real_distribution<double>  angle_prng(0.0, two_pi); // min, max
	std::normal_distribution      <double> radius_prng(0.0, 0.5);    // mean, stddev TODO should be variable

	for (int j{}; j < rounds; ++j) {
		for (size_t i{}; i < points.size(); ++i) {
			auto const  angle {  angle_prng(g) };
			auto const radius { radius_prng(g) };

			point & p { points[i] };
			point const old_pos { p };
			p += radius * point{std::sin(angle), std::cos(angle)};

			if (intersects(i, points)) p = old_pos;
		}
	}
	return init;
}


BOOST_PYTHON_MODULE(mcexercise)
{
	np::initialize();
	py::def("simulate", simulate);

	py::register_exception_translator< type_error>(static_cast<void (*)( type_error const &)>(translator));
	py::register_exception_translator<value_error>(static_cast<void (*)(value_error const &)>(translator));
}

