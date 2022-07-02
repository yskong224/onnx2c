/* This file is part of onnx2c.
 *
 * ConvTranspose node.
 * Functionality acccording to ONNX documentation:
 * "The convolution transpose operator consumes an input tensor and a filter, and computes the output."
 *
 * Pytorch opens ConvTranspose2d up a bit more:
 * "This module can be seen as the gradient of Conv2d with respect to its input. It is also known as a
 *  fractionally-strided convolution or a deconvolution (although it is not an actual deconvolution
 *  operation as it does not compute a true inverse of convolution)".
 * With added links:
 *  - https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
 *  - https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf
 * It seems to be left as an exercise for the reader to figure out if the 'Deconvolutional Networks'
 * paper describes "true inverse of convolution" or the calculations done in Convtranspose2d.
 *
 * TODO: check if Pytorch's ConvTranspose2d matches one-to-one with ONNX ConvTranspose.
 *
 * ConvTranspose is (in onnx2c) implemented as a subclass of SpatialFiter.
 */
#include "convtranspose.h"

namespace toC {


void ConvTranspose::resolve(void)
{
	// SpatialFilter expect x and w to be set to valid pointers
	x = inputs[0]; // data
	register_input(x,"x");
	w = inputs[1]; // weights
	register_input(w,"w");
	if( inputs.size() == 3 ) {
		b = inputs[2];
		register_input(b,"b");
	}
	else
		b = NULL;


	// Helper functions from SpatialFilter
	resolve_strides();
	resolve_kernel_shape();
	resolve_dilations();
	resolve_output_pads();
	resolve_convtranspose_pads();

	Tensor *rv = new Tensor;
	rv->data_dim = resolve_output_size();
	rv->data_type = x->data_type;
	register_output(rv, "y");
	y=rv;

}

void ConvTranspose::print_output_cell_init(std::ostream &dst, const std::string &y_idx) const
{
	dst << "//hello init" << std::endl;
	//dst << "y" << y_idx << "=0;" << std::endl;
}
void ConvTranspose::print_output_cell_calc(
	std::ostream &dst,
	const std::string &x_idx,
	const std::string &w_idx,
	const std::string &y_idx) const
{
	dst << "//hello calc" << std::endl;
	
	dst << "y" << y_idx; 
	dst <<    " += x" << x_idx;
	dst <<    " * w" << w_idx << ";" << std::endl;
}
void ConvTranspose::print_output_cell_finalize(std::ostream &dst, const std::string &y_idx) const
{
}
	
// Print out source for node function body
// ConvTranspose is different enough from the other spatial filters that it breaks
// the looping logic of the SpatialFilter::print_loop_with_padding_checks(dst)
void ConvTranspose::print(std::ostream &dst) const
{
	print_header_info_comment(dst);
	INDT_1 << " /* output_sizes explicitly given in ONNX model: " << (output_sizes_given?"true":"false") << " */" << std::endl;
	// from SpatialFilter. Does not work.
	//print_loop_with_padding_checks(dst);

	print_loop_transpose(dst);
}

/* Implements:
def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y

From: http://d2l.ai/chapter_computer-vision/transposed-conv.html
*/
void ConvTranspose::print_loop_transpose( std::ostream &dst) const
{
	unsigned n_data_dims = x->data_dim.size() -2;
	unsigned batch_size = x->data_dim[0];
	unsigned channels = x->data_dim[1];
	unsigned maps=y->data_dim[1];

	/* Create various indexing strings. This makes generating the loops much cleaner,
	 * and makes possible the code sharing in child classes. */
	std::string x_idx = "[b][c]";
	std::string in_kern_idxs = "[b][c]";
	std::string w_idx = "[c][m]"; // TODO: in case of groups, change c to something else
	std::string y_idx = "[b][m]";
	for( unsigned i = 0; i<n_data_dims; i++) {
	std::string i_str = std::to_string(i);
		x_idx += "[i" + i_str + "]";
		y_idx += "[o" + i_str + "]";
		in_kern_idxs += "[ii" + i_str + "]";
		w_idx += "[k" + i_str + "]";
	}

	INDT_1 << "memset(y, 0," << y->data_num_elem()*y->data_elem_size() << ");" << std::endl << std::endl;

	/* Create the loops over batches and channels.
	 * In case this SpatialFilter has a weights input (w), this first loop is over
	 * output channels (M). Othervise input channels==outputchannels, and it is named C
	 */
	INDT_1 << "for( uint32_t b=0; b<" << batch_size << "; b++ ) {" << std::endl;
	if( options.quantize ) {
		INDT_2 << "int32_t batch_min = INT32_MAX;" << std::endl;
		INDT_2 << "int32_t batch_max = INT32_MIN;" << std::endl;
		}
	if( direct_channel_map() )
		INDT_1 << "for( uint32_t m=0, c=0; m<" << maps << "; m++, c=m) {" << std::endl;
		else if( w && group > 1 ) {
		INDT_1 << "uint32_t go = " << maps/group     << "; // output group size, i.e. maps/group" << std::endl;
		INDT_1 << "uint32_t gi = " << channels/group << "; // inptput group size, i.e. channels/group" << std::endl;
		INDT_1 << "for( uint32_t g=0; g<" << group << "; g++) {" << std::endl;
			INDT_1 << "for( uint32_t m=go*g; m<go*(g+1); m++) {" << std::endl;
	}
	else
		INDT_1 << "for( uint32_t m=0; m<" << maps << "; m++) {" << std::endl;


#if 0
	// loop over outputs and inputs
	for( unsigned i = 0; i<n_data_dims; i++) {
		std::string o_idx = "o" + std::to_string(i);
		std::string i_idx = "i" + std::to_string(i);
		INDT_2 << "for( int32_t " << o_idx << "=0, ";
		   dst <<       i_idx << "=" << -pads[i] << "; ";
		   dst <<       o_idx << "<" << y->data_dim[2+i] << "; ";
		   dst <<       o_idx <<"++, "<< i_idx << "+=" << strides[i] << ") {" << std::endl;
	}
#endif
	// loop inputs
	for( unsigned i = 0; i<n_data_dims; i++) {
		std::string i_idx = "i" + std::to_string(i);
		INDT_2 << "for( int32_t " << i_idx << "=0; ";
		   dst <<       i_idx << "<" << x->data_dim[2+i] << "; ";
		   dst <<       i_idx <<"++) {" << std::endl;
	}
	print_output_cell_init(dst, y_idx);

	if (direct_channel_map())
		;
	else if( w && group > 1 )
		INDT_3 <<   "for( int32_t c=gi*g; c<gi*(g+1); c++ ) {" << std::endl;
	else    // same as above, just cleaner to read :)
		INDT_3 <<   "for( int32_t c=0; c<" << channels << "; c++ ) {" << std::endl;


	// Generate loops over outputs and kernel indices. Something like:
	// "for ( k0=0, o0=0; k0<3; k0++, o0+=1){"
	for( unsigned i = 0; i<n_data_dims; i++) {
		std::string k_idx = "k" + std::to_string(i);
		std::string o_idx = "o" + std::to_string(i);
		std::string i_idx = "i" + std::to_string(i);
		std::string o_start = i_idx + "*" + std::to_string(strides[i]) + "-" + std::to_string(pads[i]);
		std::string o_incr = o_idx + "+=" + std::to_string(dilations[i]);

		INDT_3 << "for( int32_t " << k_idx << "=0, " << o_idx << "=" << o_start << "; ";
			   dst <<       k_idx << "<" << kernel_shape[i] << "; ";
			   dst <<       k_idx <<"++, " << o_incr << ") {" << std::endl;
	}

	// check for out-of-input reading (i.e. read a pad)
	for( unsigned i = 0; i<n_data_dims; i++) {
		std::string i_str = std::to_string(i);
		INDT_4 <<  "if( o" << i_str << "<0) continue;" << std::endl;
		INDT_4 <<  "if( o" << i_str << ">=" << output_shape[i] << ") continue;" << std::endl;
#if 0
		std::string i_str = std::to_string(i);
		INDT_4 <<  "int ii" << i_str << " = i" << i_str << "+k" << i_str <<" * " << dilations[i] <<";" << std::endl;
		INDT_4 <<  "if( ii" << i_str << "<0) continue;" << std::endl;
		INDT_4 <<  "if( ii" << i_str << ">=" << x->data_dim[2+i] << ") continue;" << std::endl;
#endif
	}

	print_output_cell_calc(dst, x_idx, w_idx, y_idx);

	// close kernel loop
	for( unsigned i = 0; i<n_data_dims; i++)
		INDT_3 << "} /* k */" << std::endl;

	// close input channels loop when it is separate from output channels
	if( direct_channel_map() == false )
		INDT_3 << "} /* c */" << std::endl;
	print_output_cell_finalize(dst, y_idx);

	// close output loop
	for( unsigned i = 0; i<n_data_dims; i++)
		INDT_2 << "} /* o */" << std::endl;

	// close loops over batches and output channels
	INDT_1 << "} /* m */" << std::endl;
	if( direct_channel_map() == false && group > 1 )
		INDT_2 << "} /* g */" << std::endl;
	INDT_1 << "} /* b */" << std::endl;
}


void ConvTranspose::resolve_convtranspose_pads(void)
{

	/* The documentation says:
	 * If the pads parameter is provided the shape of the output is calculated via the following equation:
	 * output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + 
	 *                  ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]
	 */
	if( output_shape.size() == 0 )
	{
		output_sizes_given = false;
		int num_data_dim = x->rank()-2;

		// Pads most likely must have been given if output_shape isn't. But its not required...
		//resolve_pads();
		if( pads.size() == 0 )
			pads.resize(num_data_dim*2, 0);

		for(unsigned d=2; d<x->rank(); d++)
		{
			unsigned i = d-2;
			unsigned os = strides[i] * (x->data_dim[d] -1);
			os += output_padding[i];
			os += (kernel_shape[i]-1) * dilations[i] + 1;
			os -= pads[i];
			os -= pads[i+num_data_dim];

			unsigned extra_pad = (x->data_dim[d] * strides[i]) - os;

			// TODO: not sure the implicit padding needs (or should) be stored now.
			
			if( auto_pad == "SAME_UPPER" ) {
				pads[i+num_data_dim]++;
				os += extra_pad;
			}
			else if( auto_pad == "SAME_LOWER") {
				pads[i]++;
				os += extra_pad;
			}
			else
				; // Now what?
			
			output_shape.push_back(os);
		}
		// todo: split to separate function
		return;
	}

	/*
	 * output_shape can also be explicitly specified in which case pads values are auto generated using these equations:
	total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
	If (auto_pads == SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
	Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).
	*/

	output_sizes_given = true;
	int num_data_dim = x->rank()-2;
	pads.resize(num_data_dim*2);
	// loop over the width, height or 3D equivalent dimensions.
	for(unsigned d=2; d<x->rank(); d++)
	{
		unsigned i = d-2;
		//total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
		int total_padding = strides[i] * (x->data_dim[d] - 1);
		total_padding += output_padding[i];
		total_padding += (kernel_shape[i]-1)*dilations[i] + 1;
		total_padding -= output_shape[i];

		// this is needed either because:
		//  - there is a bug somewhere in onnx2c (ONNX backend tests pass with this, though :))
		//  - it is implicitly assumed in ONNX documentation
		if( total_padding < 0 ) total_padding = 0;


		pads[i] = total_padding/2;
		pads[i+num_data_dim] = total_padding/2;

		if( total_padding % 2 == 0 )
			continue;
		if( auto_pad == "SAME_UPPER" )
			pads[i+num_data_dim]++;
		else
			pads[i]++;
	}
	
}

void ConvTranspose::resolve_output_pads(void)
{
	unsigned num_data_dim = x->rank()-2;
	if( output_padding.size() == 0 )
		for( unsigned i=0; i<num_data_dim; i++)
			output_padding.push_back(0);
	
}

std::vector<int> ConvTranspose::resolve_output_size(void)
{
	std::vector<int> shape;
	// User-given values? Use those
	if( output_shape.size() != 0 ) {
		// Argh. Why are the two of different size?
		shape.push_back(x->data_dim[0]);
		shape.push_back(w->data_dim[1] * group);
		for( auto s : output_shape )
			shape.push_back(s);
		return shape;
	}
	else {

		shape.push_back(x->data_dim[0]);
		shape.push_back(x->data_dim[1]); // TODO: take groups into consideration!
		int num_data_dim = x->rank()-2;
		for(unsigned d=2; d<x->rank(); d++)
		{
			unsigned pis = d-2;
			unsigned pie = d-2+num_data_dim;
			unsigned s = x->data_dim[d] + pads[pis] + pads[pie];
			shape.push_back(s);
		}
	}
	return shape;
}
 
} // namespace

