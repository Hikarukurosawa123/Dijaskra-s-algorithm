`timescale 1ns / 1ns // `timescale time_unit/time_precision

module part1(Clock, Enable, Reset, CounterValue);

	input logic Clock, Enable, Reset;
	output logic [7:0] CounterValue;
	
	logic c1, c2, c3, c4, c5, c6, c7; 
	
	T_f u0(Clock, Enable, Reset, CounterValue[0]);
	assign c1 = Enable && CounterValue[0];
	T_f u1(Clock, c1, Reset, CounterValue[1]);
	assign c2 = c1 && CounterValue[1];
	T_f u2(Clock, c2, Reset, CounterValue[2]);
	assign c3 = c2 && CounterValue[2];
	T_f u3(Clock, c3, Reset, CounterValue[3]);
	assign c4 = c3 && CounterValue[3];
	T_f u4(Clock, c4, Reset, CounterValue[4]);
	assign c5 = c4 && CounterValue[4];
	T_f u5(Clock, c5, Reset, CounterValue[5]);
	assign c6 = c5 && CounterValue[5];
	T_f u6(Clock, c6, Reset, CounterValue[6]);
	assign c7 = c6 && CounterValue[6];
	T_f u7(Clock, c7, Reset, CounterValue[7]);
	
endmodule


module T_f(Clock, Enable, Reset, CounterValue);
	input logic Clock, Enable, Reset;
	output logic CounterValue;
	
	always_ff@(posedge Clock)
	if(Reset)
		CounterValue <= 0;

	else if((Enable && !CounterValue) || (!Enable && CounterValue))
			CounterValue <= 1;
	else
		CounterValue <= 0;
	
endmodule