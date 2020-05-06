#include <random>
#include <boost/python/numpy.hpp>

namespace py = boost::python;
namespace np = py::numpy;


class  type_error : public std::runtime_error { using runtime_error::runtime_error; };
class value_error : public std::runtime_error { using runtime_error::runtime_error; };
class shape_error : public type_error { using type_error::type_error; };

void translator( type_error const & e) { PyErr_SetString(PyExc_TypeError , e.what()); }
void translator(value_error const & e) { PyErr_SetString(PyExc_ValueError, e.what()); }


struct point {
	double x, y;
	//constexpr auto operator<=>(point const &) const noexcept = default;
};

point border;

auto simulate(py::list const box_size, np::ndarray const init, int /*rounds*/)
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

	return init;
}


BOOST_PYTHON_MODULE(mcexercise)
{
	np::initialize();
	py::def("simulate", simulate);

	py::register_exception_translator< type_error>(static_cast<void (*)( type_error const &)>(translator));
	py::register_exception_translator<value_error>(static_cast<void (*)(value_error const &)>(translator));
}

