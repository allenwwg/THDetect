#include<pybind11/pybind11.h>
#include<pybind11/stl.h>

#include"PatchManager.h"

namespace py = pybind11;
using py::arg;

PYBIND11_MODULE(acceleration, m) {

    m.doc() = "THDetection: pybind11 acceleration module";

    // Add bindings here
    m.def("foo", []() {
        return "Hello, World!";
        });

    // compute
    py::class_<boxInfo>(m,"boxInfo")
        .def_readwrite("x1", &boxInfo::x1)
        .def_readwrite("x2", &boxInfo::x2)
        .def_readwrite("y1", &boxInfo::y1)
        .def_readwrite("y2", &boxInfo::y2)
        .def_readwrite("label_id", &boxInfo::label_id)
        .def_readwrite("score", &boxInfo::score)
        .def_readwrite("isExist", &boxInfo::isExist) ;

    py::class_<PatchManager>(m, "PatchManager")
        .def(py::init<int, int, int, int>())

        .def("set_params", &PatchManager::setParams,
            arg("scale"),
            arg("step"),
            arg("img_h"),
            arg("img_w")
        )

        .def("getInfo", &PatchManager::getInfo)
        .def("getBoxCnt",&PatchManager::getBoxCnt)
        .def("add_patch", &PatchManager::AddPatch,
            arg("bboxes"),
            arg("labels"),
            arg("row_id"),
            arg("col_id")
        );
   
    

}


