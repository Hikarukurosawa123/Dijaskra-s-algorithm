`timescale 1ns / 1ns // `timescale time_unit/time_precision

//SW[2:0] data inputs
//SW[9] select signals

//LEDR[0] output display

module mux2to1(x, y, s, m);
    	input logic x;
	input logic y;
	input logic s; 
    	output logic m;

	logic c1; 	
	logic c2; 
	logic c3; 
	logic c4; 

	//expression !s
	v7404 u2(
	.pin1(s), // s input 
	.pin2(c1) //receive output 
	);

	//expression x*!s
	v7408 u1(
	.pin1(x),
	.pin2(c1),
	.pin3(c2),//testing output for the AND
	.pin4(s),
	.pin5(y),
	.pin6(c3)
	);


	//expression s*y

	v7408 u3(
	.pin1(s),//s input
	.pin2(y),//y input 
	.pin3(c3)//testing output for the AND
	);

	v7432 u4(
	.pin1(c2),//!s input
	.pin2(c3),//y input 
	.pin3(c4)//testing output for the AND
	);

	assign m = c4; 

endmodule

//create inverter 
module v7404(pin1, pin3, pin5, pin9, pin11, pin13, pin2, pin4, pin6, pin8, pin10, pin12);
	input logic pin1; 
	input logic pin3; 
	input logic pin5; 
	input logic pin9; 
	input logic pin11; 
	input logic pin13; 
	output logic pin2; 
	output logic pin4; 
	output logic pin6;
	output logic pin12;
	output logic pin10; 
	output logic pin8; 
	
	assign pin8 = ~pin9; 
	assign pin10 = ~pin11; 
	assign pin12 = ~pin13; 
	assign pin2 = ~pin1; 
	assign pin4 = ~pin3; 
	assign pin6 = ~pin5; 

endmodule 

//create four 2-input AND gates 
module v7408(pin1, pin3, pin5, pin9, pin11, pin13, pin2, pin4, pin6, pin8,
pin10, pin12);
	input logic pin1; 
	input logic pin2; 
	input logic pin4; 
	input logic pin5; 
	input logic pin8; 
	input logic pin9;
	input logic pin11; 
	input logic pin12; 
	output logic pin3;
	output logic pin6;
	output logic pin10; 
	output logic pin13;  

	assign pin3 = pin1 & pin2; 
	assign pin6 = pin4 & pin5; 
	assign pin10 = pin8 & pin9; 
	assign pin13 = pin11 & pin12; 

endmodule 

module v7432(pin1, pin3, pin5, pin9, pin11, pin13, pin2, pin4, pin6, pin8,
pin10, pin12);
	input logic pin1; 
	input logic pin2; 
	input logic pin4; 
	input logic pin5; 
	input logic pin8; 
	input logic pin9;
	input logic pin11; 
	input logic pin12; 
	output logic pin3;
	output logic pin6;
	output logic pin10; 
	output logic pin13;  

	assign pin3 = pin1 | pin2; 
	assign pin6 = pin4 | pin5; 
	assign pin10 = pin8 | pin9; 
	assign pin13 = pin11 | pin12; 

endmodule 


