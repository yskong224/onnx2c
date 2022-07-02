/* This file is part of onnx2c.
 *
 * ConvTranspose node.
 * When implementing a new node, use this template
 * as a starting point.
 *
 * This file can be kept as a single .h file with an
 * in-header implementation, or it can be split into
 * a .h and a .cc file.
 *
 * Replace all occurances of ConvTranspose in this file.
 * Some representative dummy implementation provided.
 *
 * The functions here are callbacks from the onnx2c
 * framework. See node.h for more documentation.
 */

#include "error.h"
#include "onnx.pb.h"
#include "options.h"

#include "aixlog.hpp"
#include <iostream>
#include "spatialfilter.h"
namespace toC {

class ConvTranspose : public SpatialFilter {
	public:
	ConvTranspose() {
		op_name = "ConvTranspose";
		b = NULL;
		output_sizes_given = false;
	}
	// optional inputs
	const Tensor *b;

	bool output_sizes_given;

	// Virtuals in SpatialFilter
	virtual void print_output_cell_init(std::ostream &dst, const std::string &y_idx="") const;
	virtual void print_output_cell_calc(std::ostream &dst, const std::string &x_idx="", const std::string &w_idx="", const std::string &y_idx="") const;
	virtual void print_output_cell_finalize(std::ostream &dst, const std::string &y_idx="") const;
	void print_loop_transpose(std::ostream &dst) const;

	virtual std::vector<int> resolve_output_size(void) override;

	// Virtuals in Node
	virtual void print(std::ostream &dst) const override;
	virtual void resolve(void) override;

	
	// Paddings are resolved differently on ConvTranspose than on other spatial filters
	void resolve_convtranspose_pads(void);
	void resolve_output_pads(void);
};

} // namespace

