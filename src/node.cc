
#include "error.h"
#include "graph.h"
#include "node.h"


using namespace toC;

int64_t Node::onnx_ir_version;
bool Node::is_output_N_used(unsigned N)
{
	// ONNX spec:
	// "There are two ways to leave an optional input or output unspecified:
	// the first, available only for trailing inputs and outputs, is to simply
	// not provide that input; the second method is to use an empty string in
	// place of an input or output name."

	if( (int)N >= onnx_node->output_size() )
		return false;

	if( onnx_node->output(N) == "" )
		return false;

	return true;
}

bool Node::typeConstraint_highPrecisionNumeric(const Tensor *t) const
{
	return (
		   t->data_type == onnx::TensorProto_DataType_UINT32
		|| t->data_type == onnx::TensorProto_DataType_UINT64
		|| t->data_type == onnx::TensorProto_DataType_INT32
		|| t->data_type == onnx::TensorProto_DataType_INT64
		|| t->data_type == onnx::TensorProto_DataType_FLOAT16
		|| t->data_type == onnx::TensorProto_DataType_FLOAT
		|| t->data_type == onnx::TensorProto_DataType_DOUBLE
		|| t->data_type == onnx::TensorProto_DataType_BFLOAT16
	);
}
bool Node::typeConstraint_int64(const Tensor *t) const
{
	return (
		t->data_type == onnx::TensorProto_DataType_INT64
	);
}
bool Node::typeConstraint_plainFloatingPoints(const Tensor *t) const
{
	return (
		   t->data_type == onnx::TensorProto_DataType_FLOAT16
		|| t->data_type == onnx::TensorProto_DataType_FLOAT
		|| t->data_type == onnx::TensorProto_DataType_DOUBLE
	);
}
bool Node::typeConstraint_allFloatingPoints(const Tensor *t) const
{
	return (
		   typeConstraint_plainFloatingPoints(t)
		|| t->data_type == onnx::TensorProto_DataType_BFLOAT16
	);
}
bool Node::typeConstraint_8bit(const Tensor *t) const
{
	return (
		   t->data_type == onnx::TensorProto_DataType_INT8
		|| t->data_type == onnx::TensorProto_DataType_UINT8
	);
}

bool Node::typeConstraint_integers(const Tensor *t) const
{
	return (   typeConstraint_unsigned_integers(t)
		|| typeConstraint_signed_integers(t)
	);
}

bool Node::typeConstraint_unsigned_integers(const Tensor *t) const
{
	return (
		   t->data_type == onnx::TensorProto_DataType_UINT8
		|| t->data_type == onnx::TensorProto_DataType_UINT16
		|| t->data_type == onnx::TensorProto_DataType_UINT32
		|| t->data_type == onnx::TensorProto_DataType_UINT64
	);
}
bool Node::typeConstraint_signed_integers(const Tensor *t) const
{
	return (
		   t->data_type == onnx::TensorProto_DataType_INT8
		|| t->data_type == onnx::TensorProto_DataType_INT16
		|| t->data_type == onnx::TensorProto_DataType_INT32
		|| t->data_type == onnx::TensorProto_DataType_INT64
	);
}


void Node::multidirectional_broadcast_size(
	const std::vector<int> A,
	const std::vector<int> B,
	std::vector<int> &result) const
{
		std::vector<int> dim_a = A;
		std::vector<int> dim_b = B;

		while( dim_a.size() < dim_b.size())
			dim_a.insert(dim_a.begin(), 1);
		while( dim_b.size() < dim_a.size())
			dim_b.insert(dim_b.begin(), 1);
		assert(dim_a.size() == dim_b.size());
		for( unsigned i=0; i<dim_a.size(); i++)
		{
			if( dim_a[i] == 1 || dim_b[i] == 1 )
				result.push_back( std::max(dim_a[i], dim_b[i]) );
			else if (dim_a[i] == dim_b[i])
				result.push_back( dim_a[i] );
			else
				ERROR("multidirectional_broadcast: bad tensor shapes for node " << onnx_name);
		}
}



// NB: old node implementations that dont use input_params and output_params
// must and have overridden this function.
// New mode node implementations use the Node::register_input() and
// Node::register_output() functions
void Node::print_parameters(std::ostream &dst, bool not_callsite ) const
{
	std::vector<std::string> params;

	if( not_callsite )
	{
		for( auto i : input_params ) {
			const Tensor *t = std::get<0>(i);
			std::string name = std::get<1>(i);
			params.push_back( t->print_tensor_as_const(name) );
		}
		for( auto o : output_params ) {
			const Tensor *t = std::get<0>(o);
			// A node does not know at its resolve time if an optional
			// output is used, so it registers all. Once all nodes
			// are resolved, the tensor knows if some one uses it.
			if( t->is_used() == false )
				continue;
			std::string name = std::get<1>(o);
			params.push_back( t->print_tensor(name) );
		}
	}
	else
	{
		for( auto i : input_params ) {
			const Tensor *t = std::get<0>(i);
			params.push_back( t->print_tensor_callsite() );
		}
		for( auto o : output_params ) {
			const Tensor *t = std::get<0>(o);
			if( t->is_used() == false )
				continue;
			params.push_back( t->print_tensor_callsite() );
		}
	}

	auto i = params.begin();
	dst << *i ;
	for( i++; i != params.end(); i++)
		dst << ", " << *i;
}

void Node::print_function_parameters_shapes(std::ostream &destination) const
{
	print_parameters(destination, true);
}
void Node::print_function_parameters_callsite(std::ostream &destination) const
{
	print_parameters(destination, false);
}

void Node::register_input(const Tensor *t, std::string name)
{
	input_params.push_back(function_parameter(t, name));
}
void Node::register_output(Tensor *t, std::string name)
{
	output_params.push_back(function_parameter(t, name));
	outputs.push_back(t);
}
