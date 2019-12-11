#include "processing/radontransform2d.h"
#include "simplebackprojector.h"
#include "parallelsetup.h"
#include "img/projectiondata.h"
#include <torch/extension.h>
#include <pybind11/numpy.h>
#include "rowiterator.h"

namespace py = pybind11;

using pyarray = py::array_t<float>;

template<typename T>
static constexpr T pi() { return static_cast<T>(std::atan(1.)*4.); };

template<typename T>
static constexpr T sqrt2() { return static_cast<T>(std::sqrt(2.)); };

template<typename T>
static T deg2rad(T deg) { return static_cast<T>(deg*pi<double>()/180.); };

static const pyarray& fullrange() {
	static const auto vrange { CTL::SamplingRange::linspace(0, 179, 180) };
	static const pyarray instance { 180, vrange.data() };
	return instance;
}

// calculates the 2d parallel Radon transform
pyarray radon_forward(pyarray pyimage, pyarray pytheta, bool circle, uint device = 0) {
	const auto imginfo { pyimage.request() };
	const auto thinfo { pytheta.request() };

	if (imginfo.strides[1] > imginfo.strides[0]) {
		throw std::invalid_argument("ndarray needs to be in row-major order");
	}

	const auto& hw { imginfo.shape };
	const auto ndetector {
		circle
			? std::max(hw[0], hw[1])
			: static_cast<decltype(imginfo.shape)::value_type>(std::ceil(std::max(hw[0], hw[1])*sqrt2<float>()))
	};

	const CTL::Chunk2D<float> image { static_cast<uint>(hw[1]), static_cast<uint>(hw[0]), 
		{ static_cast<float*>(imginfo.ptr), static_cast<float*>(imginfo.ptr) + imginfo.size } };
	const auto trafo { CTL::OCL::RadonTransform2D{image, device} };

	// copy pytheta buffer into an std::vector, convert to radiant
	std::vector<float> theta { static_cast<float*>(thinfo.ptr), static_cast<float*>(thinfo.ptr) + thinfo.size };
	std::for_each(theta.begin(), theta.end(), [](float& f) { f = -deg2rad(f); });

	const auto s { CTL::SamplingRange::linspace(-ndetector/2.f + .5f, ndetector/2.f - .5f, ndetector) };

	auto sino { trafo.sampleTransform(theta, s) };
	return pyarray{
		{ sino.height(), sino.width() },
		sino.rawData()
	};
}

// calculates the adjoint of radon_forward
pyarray radon_backward(pyarray pysino, pyarray pytheta, bool circle, uint device = 0) {
	const auto sinoinfo { pysino.request() };
	const auto thinfo { pytheta.request() };

	if (sinoinfo.strides[1] > sinoinfo.strides[0]) {
		throw std::invalid_argument("ndarray needs to be in row-major order");
	}

	const auto offset { circle ? sinoinfo.shape[0]/2.f : sinoinfo.shape[0]/2.f};
	const auto adj_size { circle ? uint(sinoinfo.shape[0]) : uint(sinoinfo.shape[0]/sqrt2<float>()) };

	std::vector<float> theta { static_cast<float*>(thinfo.ptr), static_cast<float*>(thinfo.ptr) + thinfo.size };
	std::for_each(theta.begin(), theta.end(), [](float& f) { f = -deg2rad(f); });

	auto trafo { CTL::SimpleBackprojector{ device } };
	trafo.configure({theta, offset});
	trafo.setVolumeSpecs(
		{adj_size, adj_size, 1},	// nbVoxels
		{1.f, 1.f, 0.f},			// voxelSize
		{0.f, 0.f, 0.f});			// offset

	auto pd { CTL::ProjectionData{static_cast<uint>(sinoinfo.shape[0]), 1, 1} };

	const auto sinoptr { static_cast<float*>(sinoinfo.ptr) };
	auto rowit { RowIterator<float,ssize_t>{sinoptr, sinoinfo.shape[0], sinoinfo.shape[1]} };
	for (auto iview {0}; iview < sinoinfo.shape[1]; iview++) {
		pd.append({{{static_cast<uint>(sinoinfo.shape[0]), 1}, { rowit.begin(iview), rowit.end(iview+1) }}});
	}

	auto adj { trafo.backproject(pd).sliceZ(0) };
	return pyarray{
		{ adj.height(), adj.width() },
		adj.rawData()
	};
}

void setKernelFileDir(const std::string& kernelFileDir) {
	CTL::OCL::OpenCLConfig::instance().setKernelFileDir(kernelFileDir);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.doc() = "2D Radon transform using CTL";
	m.def("radon_forward", &radon_forward, 
		py::arg("image"),
		py::arg("theta") = fullrange(),
		py::arg("circle") = false,
		py::arg("device") = 0);
	m.def("radon_backward", &radon_backward,
		py::arg("image"),
		py::arg("theta") = fullrange(),
		py::arg("circle") = false,
		py::arg("device") = 0);
	m.def("set_kernel_file_dir", &setKernelFileDir,
		py::arg("kernel_file_dir"));
}