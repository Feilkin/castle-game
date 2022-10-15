// Ada's port of FSR 1 to WGSL for Bevy
//
// Original copyright notice:
////////////////////////////////////////////////////////////////////////////////
// FidelityFX Super Resolution Sample
//
// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//-------------------------------------------------------------------------------

struct View {
    view_proj: mat4x4<f32>,
    inverse_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    world_position: vec3<f32>,
    width: f32,
    height: f32,
};

struct UpscaleSettings {
    upscale_factor: f32,
    sharpening_amount: f32,
}

@group(0) @binding(0)
var<uniform> view: View;

@group(1) @binding(0)
var texture: texture_2d<f32>;

@group(1) @binding(1)
var our_sampler: sampler;

@group(1) @binding(2)
var<uniform> upscale_settings: UpscaleSettings;

fn AU1_AH1_AF1_x(a: f32) -> u32{return pack2x16float(vec2(a,0.0));}

fn AAbsSU1(a: u32) -> u32 {return u32(abs(i32(a)));}
fn AAbsSU2(a: vec2<u32>) -> vec2<u32> {return vec2<u32>(abs(vec2<i32>(a)));}
fn AAbsSU3(a: vec3<u32>) -> vec3<u32> {return vec3<u32>(abs(vec3<i32>(a)));}
fn AAbsSU4(a: vec4<u32>) -> vec4<u32> {return vec4<u32>(abs(vec4<i32>(a)));}

fn ABfe(src: u32, off: u32, bits: u32) -> u32 { return extractBits(src, off, bits);}
fn ABfi(src: u32, ins: u32, mask: u32) -> u32 { return (ins&mask)|(src&(~mask));}

fn ABfiM(src: u32, ins: u32, bits: u32) -> u32 {return insertBits(src,ins,0u,bits);}

// float AClampF1(float x,float n,float m) -> f32 {return clamp(x,n,m);}
// vec2 AClampF2(vec2 x,vec2 n,vec2 m){return clamp(x,n,m);}
// vec3 AClampF3(vec3 x,vec3 n,vec3 m){return clamp(x,n,m);}
// vec4 AClampF4(vec4 x,vec4 n,vec4 m){return clamp(x,n,m);}

// vec3 AFractF3(vec3 x){return fract(x);}
// vec4 AFractF4(vec4 x){return fract(x);}

// float ALerpF1(float x,float y,float a){return mix(x,y,a);}

fn AMax3F1(x: f32, y: f32, z: f32) -> f32 {return max(x,max(y,z));}
fn AMax3F2(x: vec2<f32>, y: vec2<f32>, z: vec2<f32>) -> vec2<f32> {return max(x,max(y,z));}
fn AMax3F3(x: vec3<f32>, y: vec3<f32>, z: vec3<f32>) -> vec3<f32> {return max(x,max(y,z));}
// vec4 AMax3F4(vec4 x,vec4 y,vec4 z){return max(x,max(y,z));}

// uint AMax3SU1(uint x,uint y,uint z){return uint(max(int(x),max(int(y),int(z))));}
// uvec2 AMax3SU2(uvec2 x,uvec2 y,uvec2 z){return uvec2(max(ivec2(x),max(ivec2(y),ivec2(z))));}
// uvec3 AMax3SU3(uvec3 x,uvec3 y,uvec3 z){return uvec3(max(ivec3(x),max(ivec3(y),ivec3(z))));}
// uvec4 AMax3SU4(uvec4 x,uvec4 y,uvec4 z){return uvec4(max(ivec4(x),max(ivec4(y),ivec4(z))));}

// uint AMax3U1(uint x,uint y,uint z){return max(x,max(y,z));}
// uvec2 AMax3U2(uvec2 x,uvec2 y,uvec2 z){return max(x,max(y,z));}
// uvec3 AMax3U3(uvec3 x,uvec3 y,uvec3 z){return max(x,max(y,z));}
// uvec4 AMax3U4(uvec4 x,uvec4 y,uvec4 z){return max(x,max(y,z));}

// uint AMaxSU1(uint a,uint b){return uint(max(int(a),int(b)));}
// uvec2 AMaxSU2(uvec2 a,uvec2 b){return uvec2(max(ivec2(a),ivec2(b)));}
// uvec3 AMaxSU3(uvec3 a,uvec3 b){return uvec3(max(ivec3(a),ivec3(b)));}
// uvec4 AMaxSU4(uvec4 a,uvec4 b){return uvec4(max(ivec4(a),ivec4(b)));}

// float AMed3F1(float x,float y,float z){return max(min(x,y),min(max(x,y),z));}
// vec2 AMed3F2(vec2 x,vec2 y,vec2 z){return max(min(x,y),min(max(x,y),z));}
// vec3 AMed3F3(vec3 x,vec3 y,vec3 z){return max(min(x,y),min(max(x,y),z));}
// vec4 AMed3F4(vec4 x,vec4 y,vec4 z){return max(min(x,y),min(max(x,y),z));}

fn AMin3F1(x: f32, y: f32, z: f32) -> f32 {return min(x,min(y,z));}
// vec2 AMin3F2(vec2 x,vec2 y,vec2 z){return min(x,min(y,z));}
fn AMin3F3(x: vec3<f32>, y: vec3<f32>, z: vec3<f32>) -> vec3<f32> {return min(x,min(y,z));}
// vec4 AMin3F4(vec4 x,vec4 y,vec4 z){return min(x,min(y,z));}

// uint AMin3SU1(uint x,uint y,uint z){return uint(min(int(x),min(int(y),int(z))));}
// uvec2 AMin3SU2(uvec2 x,uvec2 y,uvec2 z){return uvec2(min(ivec2(x),min(ivec2(y),ivec2(z))));}
// uvec3 AMin3SU3(uvec3 x,uvec3 y,uvec3 z){return uvec3(min(ivec3(x),min(ivec3(y),ivec3(z))));}
// uvec4 AMin3SU4(uvec4 x,uvec4 y,uvec4 z){return uvec4(min(ivec4(x),min(ivec4(y),ivec4(z))));}

// uint AMin3U1(uint x,uint y,uint z){return min(x,min(y,z));}
// uvec2 AMin3U2(uvec2 x,uvec2 y,uvec2 z){return min(x,min(y,z));}
// uvec3 AMin3U3(uvec3 x,uvec3 y,uvec3 z){return min(x,min(y,z));}
// uvec4 AMin3U4(uvec4 x,uvec4 y,uvec4 z){return min(x,min(y,z));}

// uint AMinSU1(uint a,uint b){return uint(min(int(a),int(b)));}
// uvec2 AMinSU2(uvec2 a,uvec2 b){return uvec2(min(ivec2(a),ivec2(b)));}
// uvec3 AMinSU3(uvec3 a,uvec3 b){return uvec3(min(ivec3(a),ivec3(b)));}
// uvec4 AMinSU4(uvec4 a,uvec4 b){return uvec4(min(ivec4(a),ivec4(b)));}

// float ANCosF1(float x){return cos(xf32(float(6.28318530718)));}
// vec2 ANCosF2(vec2 x){return cos(x*vec2<f32>(float(6.28318530718)));}
// vec3 ANCosF3(vec3 x){return cos(x*vec3<f32>(float(6.28318530718)));}
// vec4 ANCosF4(vec4 x){return cos(x*vec4<f32>(float(6.28318530718)));}

// float ANSinF1(float x){return sin(xf32(float(6.28318530718)));}
// vec2 ANSinF2(vec2 x){return sin(x*vec2<f32>(float(6.28318530718)));}
// vec3 ANSinF3(vec3 x){return sin(x*vec3<f32>(float(6.28318530718)));}
// vec4 ANSinF4(vec4 x){return sin(x*vec4<f32>(float(6.28318530718)));}

fn ARcpF1(x: f32) -> f32 {return 1.0/x;}
fn ARcpF2(x: vec2<f32>) -> vec2<f32> {return vec2<f32>(1.0)/x;}
fn ARcpF3(x: vec3<f32>) -> vec3<f32> {return vec3<f32>(1.0)/x;}
fn ARcpF4(x: vec4<f32>) -> vec4<f32> {return vec4<f32>(1.0)/x;}

// float ARsqF1(float x){returnf32(float(1.0))/sqrt(x);}
// vec2 ARsqF2(vec2 x){return vec2<f32>(float(1.0))/sqrt(x);}
// vec3 ARsqF3(vec3 x){return vec3<f32>(float(1.0))/sqrt(x);}
// vec4 ARsqF4(vec4 x){return vec4<f32>(float(1.0))/sqrt(x);}

fn ASatF1(x: f32) -> f32 {return clamp(x, 0.0, 1.0);}
fn ASatF2(x: vec2<f32>) -> vec2<f32> {return clamp(x,vec2<f32>(0.0),vec2<f32>(1.0));}
fn ASatF3(x: vec3<f32>) -> vec3<f32> {return clamp(x,vec3<f32>(0.0),vec3<f32>(1.0));}
fn ASatF4(x: vec4<f32>) -> vec4<f32> {return clamp(x,vec4<f32>(0.0),vec4<f32>(1.0));}

fn AShrSU1(a: u32, b: u32) -> u32 {return u32(i32(a)>>b);}
// uvec2 AShrSU2(uvec2 a,uvec2 b){return uvec2(ivec2(a)>>ivec2(b));}
// uvec3 AShrSU3(uvec3 a,uvec3 b){return uvec3(ivec3(a)>>ivec3(b));}
// uvec4 AShrSU4(uvec4 a,uvec4 b){return uvec4(ivec4(a)>>ivec4(b));}

// float ACpySgnF1(float d,float s){return uintBitsToFloat(uint(floatBitsToUint(float(d))|(floatBitsToUint(float(s))&u32(uint(0x80000000u)))));}
// vec2 ACpySgnF2(vec2 d,vec2 s){return uintBitsToFloat(uvec2(floatBitsToUint(vec2(d))|(floatBitsToUint(vec2(s))&vec2<u32>(uint(0x80000000u)))));}
// vec3 ACpySgnF3(vec3 d,vec3 s){return uintBitsToFloat(uvec3(floatBitsToUint(vec3(d))|(floatBitsToUint(vec3(s))&vec3<u32>(uint(0x80000000u)))));}
// vec4 ACpySgnF4(vec4 d,vec4 s){return uintBitsToFloat(uvec4(floatBitsToUint(vec4(d))|(floatBitsToUint(vec4(s))&vec4<u32>(uint(0x80000000u)))));}

// float ASignedF1(float m){return ASatF1(mf32(float(uintBitsToFloat(uint(0xff800000u)))));}
// vec2 ASignedF2(vec2 m){return ASatF2(m*vec2<f32>(float(uintBitsToFloat(uint(0xff800000u)))));}
// vec3 ASignedF3(vec3 m){return ASatF3(m*vec3<f32>(float(uintBitsToFloat(uint(0xff800000u)))));}
// vec4 ASignedF4(vec4 m){return ASatF4(m*vec4<f32>(float(uintBitsToFloat(uint(0xff800000u)))));}

// float AGtZeroF1(float m){return ASatF1(mf32(float(uintBitsToFloat(uint(0x7f800000u)))));}
// vec2 AGtZeroF2(vec2 m){return ASatF2(m*vec2<f32>(float(uintBitsToFloat(uint(0x7f800000u)))));}
fn AGtZeroF3(m: vec3<f32>) -> vec3<f32> {return ASatF3(m*vec3<f32>(f32(bitcast<f32>(0x7f800000u))));}
// vec4 AGtZeroF4(vec4 m){return ASatF4(m*vec4<f32>(float(uintBitsToFloat(uint(0x7f800000u)))));}

// uint AFisToU1(uint x){return x^(( AShrSU1(x,u32(uint(31))))|u32(uint(0x80000000)));}
// uint AFisFromU1(uint x){return x^((~AShrSU1(x,u32(uint(31))))|u32(uint(0x80000000)));}

// uint AFisToHiU1(uint x){return x^(( AShrSU1(x,u32(uint(15))))|u32(uint(0x80000000)));}
// uint AFisFromHiU1(uint x){return x^((~AShrSU1(x,u32(uint(15))))|u32(uint(0x80000000)));}

// uint ABuc0ToU1(uint d,float i){return (d&0xffffff00u)|((min(uint(i),255u) )&(0x000000ffu));}
// uint ABuc1ToU1(uint d,float i){return (d&0xffff00ffu)|((min(uint(i),255u)<< 8)&(0x0000ff00u));}
// uint ABuc2ToU1(uint d,float i){return (d&0xff00ffffu)|((min(uint(i),255u)<<16)&(0x00ff0000u));}
// uint ABuc3ToU1(uint d,float i){return (d&0x00ffffffu)|((min(uint(i),255u)<<24)&(0xff000000u));}

// float ABuc0FromU1(uint i){return float((i )&255u);}
// float ABuc1FromU1(uint i){return float((i>> 8)&255u);}
// float ABuc2FromU1(uint i){return float((i>>16)&255u);}
// float ABuc3FromU1(uint i){return float((i>>24)&255u);}

// uint ABsc0ToU1(uint d,float i){return (d&0xffffff00u)|((min(uint(i+128.0),255u) )&(0x000000ffu));}
// uint ABsc1ToU1(uint d,float i){return (d&0xffff00ffu)|((min(uint(i+128.0),255u)<< 8)&(0x0000ff00u));}
// uint ABsc2ToU1(uint d,float i){return (d&0xff00ffffu)|((min(uint(i+128.0),255u)<<16)&(0x00ff0000u));}
// uint ABsc3ToU1(uint d,float i){return (d&0x00ffffffu)|((min(uint(i+128.0),255u)<<24)&(0xff000000u));}

// uint ABsc0ToZbU1(uint d,float i){return ((d&0xffffff00u)|((min(uint(trunc(i)+128.0),255u) )&(0x000000ffu)))^0x00000080u;}
// uint ABsc1ToZbU1(uint d,float i){return ((d&0xffff00ffu)|((min(uint(trunc(i)+128.0),255u)<< 8)&(0x0000ff00u)))^0x00008000u;}
// uint ABsc2ToZbU1(uint d,float i){return ((d&0xff00ffffu)|((min(uint(trunc(i)+128.0),255u)<<16)&(0x00ff0000u)))^0x00800000u;}
// uint ABsc3ToZbU1(uint d,float i){return ((d&0x00ffffffu)|((min(uint(trunc(i)+128.0),255u)<<24)&(0xff000000u)))^0x80000000u;}

// float ABsc0FromU1(uint i){return float((i )&255u)-128.0;}
// float ABsc1FromU1(uint i){return float((i>> 8)&255u)-128.0;}
// float ABsc2FromU1(uint i){return float((i>>16)&255u)-128.0;}
// float ABsc3FromU1(uint i){return float((i>>24)&255u)-128.0;}

// float ABsc0FromZbU1(uint i){return float(((i )&255u)^0x80u)-128.0;}
// float ABsc1FromZbU1(uint i){return float(((i>> 8)&255u)^0x80u)-128.0;}
// float ABsc2FromZbU1(uint i){return float(((i>>16)&255u)^0x80u)-128.0;}
// float ABsc3FromZbU1(uint i){return float(((i>>24)&255u)^0x80u)-128.0;}

// float APrxLoSqrtF1(float a){return uintBitsToFloat(uint((floatBitsToUint(float(a))>>u32(uint(1)))+u32(uint(0x1fbc4639))));}
fn APrxLoRcpF1(a: f32) -> f32 {return bitcast<f32>(u32(u32(u32(0x7ef07ebb))-bitcast<u32>(f32(a))));}
fn APrxMedRcpF1(a: f32) -> f32 {let b=bitcast<f32>(u32(u32(u32(0x7ef19fff))-bitcast<u32>(f32(a))));return b*(-b*a+2.0);}
fn APrxLoRsqF1(a: f32) -> f32 {return bitcast<f32>(u32(u32(u32(0x5f347d74))-(bitcast<u32>(f32(a))>>u32(u32(1)))));}

fn APrxLoSqrtF2(a: vec2<f32>) -> vec2<f32> {return bitcast<f32>(vec2<u32>((bitcast<u32>(vec2(a))>>vec2<u32>(u32(1)))+vec2<u32>(u32(0x1fbc4639))));}
// vec2 APrxLoRcpF2(vec2 a){return uintBitsToFloat(uvec2(vec2<u32>(uint(0x7ef07ebb))-floatBitsToUint(vec2(a))));}
// vec2 APrxMedRcpF2(vec2 a){vec2 b=uintBitsToFloat(uvec2(vec2<u32>(uint(0x7ef19fff))-floatBitsToUint(vec2(a))));return b*(-b*a+vec2<f32>(float(2.0)));}
// vec2 APrxLoRsqF2(vec2 a){return uintBitsToFloat(uvec2(vec2<u32>(uint(0x5f347d74))-(floatBitsToUint(vec2(a))>>vec2<u32>(uint(1)))));}

// vec3 APrxLoSqrtF3(vec3 a){return uintBitsToFloat(uvec3((floatBitsToUint(vec3(a))>>vec3<u32>(uint(1)))+vec3<u32>(uint(0x1fbc4639))));}
// vec3 APrxLoRcpF3(vec3 a){return uintBitsToFloat(uvec3(vec3<u32>(uint(0x7ef07ebb))-floatBitsToUint(vec3(a))));}
fn APrxMedRcpF3(a: vec3<f32>) -> vec3<f32> {let b=bitcast<f32>(vec3<u32>(vec3<u32>(u32(0x7ef19fff))-bitcast<u32>(vec3<f32>(a))));return b*(-b*a+vec3<f32>(f32(2.0)));}
// vec3 APrxLoRsqF3(vec3 a){return uintBitsToFloat(uvec3(vec3<u32>(uint(0x5f347d74))-(floatBitsToUint(vec3(a))>>vec3<u32>(uint(1)))));}

// vec4 APrxLoSqrtF4(vec4 a){return uintBitsToFloat(uvec4((floatBitsToUint(vec4(a))>>vec4<u32>(uint(1)))+vec4<u32>(uint(0x1fbc4639))));}
// vec4 APrxLoRcpF4(vec4 a){return uintBitsToFloat(uvec4(vec4<u32>(uint(0x7ef07ebb))-floatBitsToUint(vec4(a))));}
// vec4 APrxMedRcpF4(vec4 a){vec4 b=uintBitsToFloat(uvec4(vec4<u32>(uint(0x7ef19fff))-floatBitsToUint(vec4(a))));return b*(-b*a+vec4<f32>(float(2.0)));}
// vec4 APrxLoRsqF4(vec4 a){return uintBitsToFloat(uvec4(vec4<u32>(uint(0x5f347d74))-(floatBitsToUint(vec4(a))>>vec4<u32>(uint(1)))));}

// TODO: figure out this mess
//fn Quart_f32(a: f32) -> f32 { a = a * a; return a * a;}
//fn Oct_f32(a: f32) -> f32 { a = a * a; a = a * a; return a * a; }
//fn Quart(a: vec2<f32>) -> vec2<f32> { a = a * a; return a * a; }
// vec2 Oct(vec2 a) { a = a * a; a = a * a; return a * a; }
// vec3 Quart(vec3 a) { a = a * a; return a * a; }
// vec3 Oct(vec3 a) { a = a * a; a = a * a; return a * a; }
// vec4 Quart(vec4 a) { a = a * a; return a * a; }
// vec4 Oct(vec4 a) { a = a * a; a = a * a; return a * a; }
//
//fn APrxPQToGamma2(a: f32) -> f32 { return Quart_f32(a); }
//fn APrxPQToLinear(a: f32) -> f32 { return Oct_f32(a); }
// float APrxLoGamma2ToPQ(float a) { return uintBitsToFloat(uint((floatBitsToUint(float(a)) >> u32(uint(2))) + u32(uint(0x2F9A4E46)))); }
// float APrxMedGamma2ToPQ(float a) { float b = uintBitsToFloat(uint((floatBitsToUint(float(a)) >> u32(uint(2))) + u32(uint(0x2F9A4E46)))); float b4 = Quart(b); return b - b * (b4 - a) / f32(float(4.0)) * b4); }
// float APrxHighGamma2ToPQ(float a) { return sqrt(sqrt(a)); }
// float APrxLoLinearToPQ(float a) { return uintBitsToFloat(uint((floatBitsToUint(float(a)) >> u32(uint(3))) + u32(uint(0x378D8723)))); }
// float APrxMedLinearToPQ(float a) { float b = uintBitsToFloat(uint((floatBitsToUint(float(a)) >> u32(uint(3))) + u32(uint(0x378D8723)))); float b8 = Oct(b); return b - b * (b8 - a) / f32(float(8.0)) * b8); }
// float APrxHighLinearToPQ(float a) { return sqrt(sqrt(sqrt(a))); }
//
// vec2 APrxPQToGamma2(vec2 a) { return Quart(a); }
// vec2 APrxPQToLinear(vec2 a) { return Oct(a); }
// vec2 APrxLoGamma2ToPQ(vec2 a) { return uintBitsToFloat(uvec2((floatBitsToUint(vec2(a)) >> vec2<u32>(uint(2))) + vec2<u32>(uint(0x2F9A4E46)))); }
// vec2 APrxMedGamma2ToPQ(vec2 a) { vec2 b = uintBitsToFloat(uvec2((floatBitsToUint(vec2(a)) >> vec2<u32>(uint(2))) + vec2<u32>(uint(0x2F9A4E46)))); vec2 b4 = Quart(b); return b - b * (b4 - a) / f32(float(4.0)) * b4); }
// vec2 APrxHighGamma2ToPQ(vec2 a) { return sqrt(sqrt(a)); }
// vec2 APrxLoLinearToPQ(vec2 a) { return uintBitsToFloat(uvec2((floatBitsToUint(vec2(a)) >> vec2<u32>(uint(3))) + vec2<u32>(uint(0x378D8723)))); }
// vec2 APrxMedLinearToPQ(vec2 a) { vec2 b = uintBitsToFloat(uvec2((floatBitsToUint(vec2(a)) >> vec2<u32>(uint(3))) + vec2<u32>(uint(0x378D8723)))); vec2 b8 = Oct(b); return b - b * (b8 - a) / f32(float(8.0)) * b8); }
// vec2 APrxHighLinearToPQ(vec2 a) { return sqrt(sqrt(sqrt(a))); }
//
// vec3 APrxPQToGamma2(vec3 a) { return Quart(a); }
// vec3 APrxPQToLinear(vec3 a) { return Oct(a); }
// vec3 APrxLoGamma2ToPQ(vec3 a) { return uintBitsToFloat(uvec3((floatBitsToUint(vec3(a)) >> vec3<u32>(uint(2))) + vec3<u32>(uint(0x2F9A4E46)))); }
// vec3 APrxMedGamma2ToPQ(vec3 a) { vec3 b = uintBitsToFloat(uvec3((floatBitsToUint(vec3(a)) >> vec3<u32>(uint(2))) + vec3<u32>(uint(0x2F9A4E46)))); vec3 b4 = Quart(b); return b - b * (b4 - a) / f32(float(4.0)) * b4); }
// vec3 APrxHighGamma2ToPQ(vec3 a) { return sqrt(sqrt(a)); }
// vec3 APrxLoLinearToPQ(vec3 a) { return uintBitsToFloat(uvec3((floatBitsToUint(vec3(a)) >> vec3<u32>(uint(3))) + vec3<u32>(uint(0x378D8723)))); }
// vec3 APrxMedLinearToPQ(vec3 a) { vec3 b = uintBitsToFloat(uvec3((floatBitsToUint(vec3(a)) >> vec3<u32>(uint(3))) + vec3<u32>(uint(0x378D8723)))); vec3 b8 = Oct(b); return b - b * (b8 - a) / f32(float(8.0)) * b8); }
// vec3 APrxHighLinearToPQ(vec3 a) { return sqrt(sqrt(sqrt(a))); }
//
// vec4 APrxPQToGamma2(vec4 a) { return Quart(a); }
// vec4 APrxPQToLinear(vec4 a) { return Oct(a); }
// vec4 APrxLoGamma2ToPQ(vec4 a) { return uintBitsToFloat(uvec4((floatBitsToUint(vec4(a)) >> vec4<u32>(uint(2))) + vec4<u32>(uint(0x2F9A4E46)))); }
// vec4 APrxMedGamma2ToPQ(vec4 a) { vec4 b = uintBitsToFloat(uvec4((floatBitsToUint(vec4(a)) >> vec4<u32>(uint(2))) + vec4<u32>(uint(0x2F9A4E46)))); vec4 b4 = Quart(b); return b - b * (b4 - a) / f32(float(4.0)) * b4); }
// vec4 APrxHighGamma2ToPQ(vec4 a) { return sqrt(sqrt(a)); }
// vec4 APrxLoLinearToPQ(vec4 a) { return uintBitsToFloat(uvec4((floatBitsToUint(vec4(a)) >> vec4<u32>(uint(3))) + vec4<u32>(uint(0x378D8723)))); }
// vec4 APrxMedLinearToPQ(vec4 a) { vec4 b = uintBitsToFloat(uvec4((floatBitsToUint(vec4(a)) >> vec4<u32>(uint(3))) + vec4<u32>(uint(0x378D8723)))); vec4 b8 = Oct(b); return b - b * (b8 - a) / f32(float(8.0)) * b8); }
// vec4 APrxHighLinearToPQ(vec4 a) { return sqrt(sqrt(sqrt(a))); }

fn APSinF1(x: f32) -> f32 {return x*abs(x)-x;}
fn APSinF2(x: vec2<f32>) -> vec2<f32> {return x*abs(x)-x;}
// float APCosF1(float x){x=fract(xf32(float(0.5))f32(float(0.75)));x=xf32(float(2.0))f32(float(1.0));return APSinF1(x);}
// vec2 APCosF2(vec2 x){x=fract(x*vec2<f32>(float(0.5))+vec2<f32>(float(0.75)));x=x*vec2<f32>(float(2.0))-vec2<f32>(float(1.0));return APSinF2(x);}
// vec2 APSinCosF1(float x){float y=fract(xf32(float(0.5))f32(float(0.75)));y=yf32(float(2.0))f32(float(1.0));return APSinF2(vec2(x,y));}

// uint AZolAndU1(uint x,uint y){return min(x,y);}
// uvec2 AZolAndU2(uvec2 x,uvec2 y){return min(x,y);}
// uvec3 AZolAndU3(uvec3 x,uvec3 y){return min(x,y);}
// uvec4 AZolAndU4(uvec4 x,uvec4 y){return min(x,y);}

// uint AZolNotU1(uint x){return x^u32(uint(1));}
// uvec2 AZolNotU2(uvec2 x){return x^vec2<u32>(uint(1));}
// uvec3 AZolNotU3(uvec3 x){return x^vec3<u32>(uint(1));}
// uvec4 AZolNotU4(uvec4 x){return x^vec4<u32>(uint(1));}

// uint AZolOrU1(uint x,uint y){return max(x,y);}
// uvec2 AZolOrU2(uvec2 x,uvec2 y){return max(x,y);}
// uvec3 AZolOrU3(uvec3 x,uvec3 y){return max(x,y);}
// uvec4 AZolOrU4(uvec4 x,uvec4 y){return max(x,y);}

// uint AZolF1ToU1(float x){return uint(x);}
// uvec2 AZolF2ToU2(vec2 x){return uvec2(x);}
// uvec3 AZolF3ToU3(vec3 x){return uvec3(x);}
// uvec4 AZolF4ToU4(vec4 x){return uvec4(x);}

// uint AZolNotF1ToU1(float x){return uintf32(float(1.0))-x);}
// uvec2 AZolNotF2ToU2(vec2 x){return uvec2(vec2<f32>(float(1.0))-x);}
// uvec3 AZolNotF3ToU3(vec3 x){return uvec3(vec3<f32>(float(1.0))-x);}
// uvec4 AZolNotF4ToU4(vec4 x){return uvec4(vec4<f32>(float(1.0))-x);}

// float AZolU1ToF1(uint x){return float(x);}
// vec2 AZolU2ToF2(uvec2 x){return vec2(x);}
// vec3 AZolU3ToF3(uvec3 x){return vec3(x);}
// vec4 AZolU4ToF4(uvec4 x){return vec4(x);}

// float AZolAndF1(float x,float y){return min(x,y);}
// vec2 AZolAndF2(vec2 x,vec2 y){return min(x,y);}
// vec3 AZolAndF3(vec3 x,vec3 y){return min(x,y);}
// vec4 AZolAndF4(vec4 x,vec4 y){return min(x,y);}

// float ASolAndNotF1(float x,float y){return (-x)*yf32(float(1.0));}
// vec2 ASolAndNotF2(vec2 x,vec2 y){return (-x)*y+vec2<f32>(float(1.0));}
// vec3 ASolAndNotF3(vec3 x,vec3 y){return (-x)*y+vec3<f32>(float(1.0));}
// vec4 ASolAndNotF4(vec4 x,vec4 y){return (-x)*y+vec4<f32>(float(1.0));}

// float AZolAndOrF1(float x,float y,float z){return ASatF1(x*y+z);}
// vec2 AZolAndOrF2(vec2 x,vec2 y,vec2 z){return ASatF2(x*y+z);}
// vec3 AZolAndOrF3(vec3 x,vec3 y,vec3 z){return ASatF3(x*y+z);}
// vec4 AZolAndOrF4(vec4 x,vec4 y,vec4 z){return ASatF4(x*y+z);}

// float AZolGtZeroF1(float x){return ASatF1(xf32(float(uintBitsToFloat(uint(0x7f800000u)))));}
// vec2 AZolGtZeroF2(vec2 x){return ASatF2(x*vec2<f32>(float(uintBitsToFloat(uint(0x7f800000u)))));}
// vec3 AZolGtZeroF3(vec3 x){return ASatF3(x*vec3<f32>(float(uintBitsToFloat(uint(0x7f800000u)))));}
// vec4 AZolGtZeroF4(vec4 x){return ASatF4(x*vec4<f32>(float(uintBitsToFloat(uint(0x7f800000u)))));}

// float AZolNotF1(float x){returnf32(float(1.0))-x;}
// vec2 AZolNotF2(vec2 x){return vec2<f32>(float(1.0))-x;}
// vec3 AZolNotF3(vec3 x){return vec3<f32>(float(1.0))-x;}
// vec4 AZolNotF4(vec4 x){return vec4<f32>(float(1.0))-x;}

// float AZolOrF1(float x,float y){return max(x,y);}
// vec2 AZolOrF2(vec2 x,vec2 y){return max(x,y);}
// vec3 AZolOrF3(vec3 x,vec3 y){return max(x,y);}
// vec4 AZolOrF4(vec4 x,vec4 y){return max(x,y);}

fn AZolSelF1(x: f32, y: f32, z: f32) -> f32 {let r=(-x)*z+z;return x*y+r;}
fn AZolSelF2(x: vec2<f32>, y: vec2<f32>, z: vec2<f32>) -> vec2<f32> {let r=(-x)*z+z;return x*y+r;}
fn AZolSelF3(x: vec3<f32>, y: vec3<f32>, z: vec3<f32>) -> vec3<f32> {let r=(-x)*z+z;return x*y+r;}
fn AZolSelF4(x: vec4<f32>, y: vec4<f32>, z: vec4<f32>) -> vec4<f32> {let r=(-x)*z+z;return x*y+r;}

fn AZolSignedF1(x: f32) -> f32 {return ASatF1(x*bitcast<f32>(0xff800000u));}
fn AZolSignedF2(x: vec2<f32>) -> vec2<f32> {return ASatF2(x*vec2<f32>(bitcast<f32>(0xff800000u)));}
fn AZolSignedF3(x: vec3<f32>) -> vec3<f32> {return ASatF3(x*vec3<f32>(bitcast<f32>(0xff800000u)));}
fn AZolSignedF4(x: vec4<f32>) -> vec4<f32> {return ASatF4(x*vec4<f32>(bitcast<f32>(0xff800000u)));}

// float AZolZeroPassF1(float x,float y){return uintBitsToFloat(uint((floatBitsToUint(float(x))!=u32(uint(0)))?u32(uint(0)):floatBitsToUint(float(y))));}
// vec2 AZolZeroPassF2(vec2 x,vec2 y){return uintBitsToFloat(uvec2((floatBitsToUint(vec2(x))!=vec2<u32>(uint(0)))?vec2<u32>(uint(0)):floatBitsToUint(vec2(y))));}
// vec3 AZolZeroPassF3(vec3 x,vec3 y){return uintBitsToFloat(uvec3((floatBitsToUint(vec3(x))!=vec3<u32>(uint(0)))?vec3<u32>(uint(0)):floatBitsToUint(vec3(y))));}
// vec4 AZolZeroPassF4(vec4 x,vec4 y){return uintBitsToFloat(uvec4((floatBitsToUint(vec4(x))!=vec4<u32>(uint(0)))?vec4<u32>(uint(0)):floatBitsToUint(vec4(y))));}

// float ATo709F1(float c){vec3 j=vec3(0.018*4.5,4.5,0.45);vec2 k=vec2(1.099,-0.099);
//     return clamp(j.x ,c*j.y ,pow(c,j.z )*k.x +k.y );}
// vec2 ATo709F2(vec2 c){vec3 j=vec3(0.018*4.5,4.5,0.45);vec2 k=vec2(1.099,-0.099);
//     return clamp(j.xx ,c*j.yy ,pow(c,j.zz )*k.xx +k.yy );}
// vec3 ATo709F3(vec3 c){vec3 j=vec3(0.018*4.5,4.5,0.45);vec2 k=vec2(1.099,-0.099);
//     return clamp(j.xxx,c*j.yyy,pow(c,j.zzz)*k.xxx+k.yyy);}

// float AToGammaF1(float c,float rcpX){return pow(cf32(float(rcpX)));}
// vec2 AToGammaF2(vec2 c,float rcpX){return pow(c,vec2<f32>(float(rcpX)));}
// vec3 AToGammaF3(vec3 c,float rcpX){return pow(c,vec3<f32>(float(rcpX)));}

// float AToPqF1(float x){float p=pow(xf32(float(0.159302)));
//     return pow(f32(float(0.835938))f32(float(18.8516))*p)/f32(float(1.0))f32(float(18.6875))*p)f32(float(78.8438)));}
// vec2 AToPqF1(vec2 x){vec2 p=pow(x,vec2<f32>(float(0.159302)));
//     return pow((vec2<f32>(float(0.835938))+vec2<f32>(float(18.8516))*p)/(vec2<f32>(float(1.0))+vec2<f32>(float(18.6875))*p),vec2<f32>(float(78.8438)));}
// vec3 AToPqF1(vec3 x){vec3 p=pow(x,vec3<f32>(float(0.159302)));
//     return pow((vec3<f32>(float(0.835938))+vec3<f32>(float(18.8516))*p)/(vec3<f32>(float(1.0))+vec3<f32>(float(18.6875))*p),vec3<f32>(float(78.8438)));}

// float AToSrgbF1(float c){vec3 j=vec3(0.0031308*12.92,12.92,1.0/2.4);vec2 k=vec2(1.055,-0.055);
//     return clamp(j.x ,c*j.y ,pow(c,j.z )*k.x +k.y );}
// vec2 AToSrgbF2(vec2 c){vec3 j=vec3(0.0031308*12.92,12.92,1.0/2.4);vec2 k=vec2(1.055,-0.055);
//     return clamp(j.xx ,c*j.yy ,pow(c,j.zz )*k.xx +k.yy );}
// vec3 AToSrgbF3(vec3 c){vec3 j=vec3(0.0031308*12.92,12.92,1.0/2.4);vec2 k=vec2(1.055,-0.055);
//     return clamp(j.xxx,c*j.yyy,pow(c,j.zzz)*k.xxx+k.yyy);}

// float AToTwoF1(float c){return sqrt(c);}
// vec2 AToTwoF2(vec2 c){return sqrt(c);}
// vec3 AToTwoF3(vec3 c){return sqrt(c);}

// float AToThreeF1(float c){return pow(cf32(float(1.0/3.0)));}
// vec2 AToThreeF2(vec2 c){return pow(c,vec2<f32>(float(1.0/3.0)));}
// vec3 AToThreeF3(vec3 c){return pow(c,vec3<f32>(float(1.0/3.0)));}

// float AFrom709F1(float c){vec3 j=vec3(0.081/4.5,1.0/4.5,1.0/0.45);vec2 k=vec2(1.0/1.099,0.099/1.099);
//     return AZolSelF1(AZolSignedF1(c-j.x ),c*j.y ,pow(c*k.x +k.y ,j.z ));}
// vec2 AFrom709F2(vec2 c){vec3 j=vec3(0.081/4.5,1.0/4.5,1.0/0.45);vec2 k=vec2(1.0/1.099,0.099/1.099);
//     return AZolSelF2(AZolSignedF2(c-j.xx ),c*j.yy ,pow(c*k.xx +k.yy ,j.zz ));}
// vec3 AFrom709F3(vec3 c){vec3 j=vec3(0.081/4.5,1.0/4.5,1.0/0.45);vec2 k=vec2(1.0/1.099,0.099/1.099);
//     return AZolSelF3(AZolSignedF3(c-j.xxx),c*j.yyy,pow(c*k.xxx+k.yyy,j.zzz));}

// float AFromGammaF1(float c,float x){return pow(cf32(float(x)));}
// vec2 AFromGammaF2(vec2 c,float x){return pow(c,vec2<f32>(float(x)));}
// vec3 AFromGammaF3(vec3 c,float x){return pow(c,vec3<f32>(float(x)));}

// float AFromPqF1(float x){float p=pow(xf32(float(0.0126833)));
//     return pow(ASatF1(pf32(float(0.835938)))/f32(float(18.8516))f32(float(18.6875))*p)f32(float(6.27739)));}
// vec2 AFromPqF1(vec2 x){vec2 p=pow(x,vec2<f32>(float(0.0126833)));
//     return pow(ASatF2(p-vec2<f32>(float(0.835938)))/(vec2<f32>(float(18.8516))-vec2<f32>(float(18.6875))*p),vec2<f32>(float(6.27739)));}
// vec3 AFromPqF1(vec3 x){vec3 p=pow(x,vec3<f32>(float(0.0126833)));
//     return pow(ASatF3(p-vec3<f32>(float(0.835938)))/(vec3<f32>(float(18.8516))-vec3<f32>(float(18.6875))*p),vec3<f32>(float(6.27739)));}

// float AFromSrgbF1(float c){vec3 j=vec3(0.04045/12.92,1.0/12.92,2.4);vec2 k=vec2(1.0/1.055,0.055/1.055);
//     return AZolSelF1(AZolSignedF1(c-j.x ),c*j.y ,pow(c*k.x +k.y ,j.z ));}
// vec2 AFromSrgbF2(vec2 c){vec3 j=vec3(0.04045/12.92,1.0/12.92,2.4);vec2 k=vec2(1.0/1.055,0.055/1.055);
//     return AZolSelF2(AZolSignedF2(c-j.xx ),c*j.yy ,pow(c*k.xx +k.yy ,j.zz ));}
// vec3 AFromSrgbF3(vec3 c){vec3 j=vec3(0.04045/12.92,1.0/12.92,2.4);vec2 k=vec2(1.0/1.055,0.055/1.055);
//     return AZolSelF3(AZolSignedF3(c-j.xxx),c*j.yyy,pow(c*k.xxx+k.yyy,j.zzz));}

// float AFromTwoF1(float c){return c*c;}
// vec2 AFromTwoF2(vec2 c){return c*c;}
// vec3 AFromTwoF3(vec3 c){return c*c;}
//
// float AFromThreeF1(float c){return c*c*c;}
// vec2 AFromThreeF2(vec2 c){return c*c*c;}
// vec3 AFromThreeF3(vec3 c){return c*c*c;}

// uvec2 ARmp8x8(uint a){return uvec2(ABfe(a,1u,3u),ABfiM(ABfe(a,3u,3u),a,1u));}

// uvec2 ARmpRed8x8(uint a){return uvec2(ABfiM(ABfe(a,2u,3u),a,1u),ABfiM(ABfe(a,3u,3u),ABfe(a,1u,2u),2u));}

// vec2 opAAbsF2(out vec2 d,in vec2 a){d=abs(a);return d;}
// vec3 opAAbsF3(out vec3 d,in vec3 a){d=abs(a);return d;}
// vec4 opAAbsF4(out vec4 d,in vec4 a){d=abs(a);return d;}

// vec2 opAAddF2(out vec2 d,in vec2 a,in vec2 b){d=a+b;return d;}
// vec3 opAAddF3(out vec3 d,in vec3 a,in vec3 b){d=a+b;return d;}
// vec4 opAAddF4(out vec4 d,in vec4 a,in vec4 b){d=a+b;return d;}

// vec2 opAAddOneF2(out vec2 d,in vec2 a,float b){d=a+vec2<f32>(float(b));return d;}
// vec3 opAAddOneF3(out vec3 d,in vec3 a,float b){d=a+vec3<f32>(float(b));return d;}
// vec4 opAAddOneF4(out vec4 d,in vec4 a,float b){d=a+vec4<f32>(float(b));return d;}

// vec2 opACpyF2(out vec2 d,in vec2 a){d=a;return d;}
// vec3 opACpyF3(out vec3 d,in vec3 a){d=a;return d;}
// vec4 opACpyF4(out vec4 d,in vec4 a){d=a;return d;}

// vec2 opmix(out vec2 d,in vec2 a,in vec2 b,in vec2 c){d=mix(a,b,c);return d;}
// vec3 opmix(out vec3 d,in vec3 a,in vec3 b,in vec3 c){d=mix(a,b,c);return d;}
// vec4 opmix(out vec4 d,in vec4 a,in vec4 b,in vec4 c){d=mix(a,b,c);return d;}

// vec2 opALerpOneF2(out vec2 d,in vec2 a,in vec2 b,float c){d=mix(a,b,vec2<f32>(float(c)));return d;}
// vec3 opALerpOneF3(out vec3 d,in vec3 a,in vec3 b,float c){d=mix(a,b,vec3<f32>(float(c)));return d;}
// vec4 opALerpOneF4(out vec4 d,in vec4 a,in vec4 b,float c){d=mix(a,b,vec4<f32>(float(c)));return d;}

// vec2 opAMaxF2(out vec2 d,in vec2 a,in vec2 b){d=max(a,b);return d;}
// vec3 opAMaxF3(out vec3 d,in vec3 a,in vec3 b){d=max(a,b);return d;}
// vec4 opAMaxF4(out vec4 d,in vec4 a,in vec4 b){d=max(a,b);return d;}

// vec2 opAMinF2(out vec2 d,in vec2 a,in vec2 b){d=min(a,b);return d;}
// vec3 opAMinF3(out vec3 d,in vec3 a,in vec3 b){d=min(a,b);return d;}
// vec4 opAMinF4(out vec4 d,in vec4 a,in vec4 b){d=min(a,b);return d;}

// vec2 opAMulF2(out vec2 d,in vec2 a,in vec2 b){d=a*b;return d;}
// vec3 opAMulF3(out vec3 d,in vec3 a,in vec3 b){d=a*b;return d;}
// vec4 opAMulF4(out vec4 d,in vec4 a,in vec4 b){d=a*b;return d;}
//
// vec2 opAMulOneF2(out vec2 d,in vec2 a,float b){d=a*vec2<f32>(float(b));return d;}
// vec3 opAMulOneF3(out vec3 d,in vec3 a,float b){d=a*vec3<f32>(float(b));return d;}
// vec4 opAMulOneF4(out vec4 d,in vec4 a,float b){d=a*vec4<f32>(float(b));return d;}
//
// vec2 opANegF2(out vec2 d,in vec2 a){d=-a;return d;}
// vec3 opANegF3(out vec3 d,in vec3 a){d=-a;return d;}
// vec4 opANegF4(out vec4 d,in vec4 a){d=-a;return d;}
//
// vec2 opARcpF2(out vec2 d,in vec2 a){d=ARcpF2(a);return d;}
// vec3 opARcpF3(out vec3 d,in vec3 a){d=ARcpF3(a);return d;}
// vec4 opARcpF4(out vec4 d,in vec4 a){d=ARcpF4(a);return d;}

fn FsrEasuRF(p: vec2<f32>) -> vec4<f32> {return textureGather(0, texture, our_sampler, p);}
fn FsrEasuGF(p: vec2<f32>) -> vec4<f32> {return textureGather(1, texture, our_sampler, p);}
fn FsrEasuBF(p: vec2<f32>) -> vec4<f32> {return textureGather(2, texture, our_sampler, p);}

fn FsrEasuCon(
    con0: ptr<function, vec4<u32>>,
    con1: ptr<function, vec4<u32>>,
    con2: ptr<function, vec4<u32>>,
    con3: ptr<function, vec4<u32>>,

    inputViewportInPixelsX: f32,
    inputViewportInPixelsY: f32,

    inputSizeInPixelsX: f32,
    inputSizeInPixelsY: f32,

    outputSizeInPixelsX: f32,
    outputSizeInPixelsY: f32
) {

     (*con0)[0]=bitcast<u32>(f32(inputViewportInPixelsX*ARcpF1(outputSizeInPixelsX)));
     (*con0)[1]=bitcast<u32>(f32(inputViewportInPixelsY*ARcpF1(outputSizeInPixelsY)));
     (*con0)[2]=bitcast<u32>(0.5*inputViewportInPixelsX*ARcpF1(outputSizeInPixelsX)- 0.5);
     (*con0)[3]=bitcast<u32>(0.5*inputViewportInPixelsY*ARcpF1(outputSizeInPixelsY)- 0.5);

     (*con1)[0]=bitcast<u32>(f32(ARcpF1(inputSizeInPixelsX)));
     (*con1)[1]=bitcast<u32>(f32(ARcpF1(inputSizeInPixelsY)));

     (*con1)[2]=bitcast<u32>(1.0*ARcpF1(inputSizeInPixelsX));
     (*con1)[3]=bitcast<u32>(-1.0*ARcpF1(inputSizeInPixelsY));

     (*con2)[0]=bitcast<u32>(-1.0*ARcpF1(inputSizeInPixelsX));
     (*con2)[1]=bitcast<u32>(2.0*ARcpF1(inputSizeInPixelsY));
     (*con2)[2]=bitcast<u32>(1.0*ARcpF1(inputSizeInPixelsX));
     (*con2)[3]=bitcast<u32>(2.0*ARcpF1(inputSizeInPixelsY));
     (*con3)[0]=bitcast<u32>(0.0*ARcpF1(inputSizeInPixelsX));
     (*con3)[1]=bitcast<u32>(4.0*ARcpF1(inputSizeInPixelsY));
     (*con3)[2]=0u;
     (*con3)[3]=0u;}

//fn FsrEasuConOffset(
// con0: vec4<u32>,
// con1: vec4<u32>,
// con2: vec4<u32>,
// con3: vec4<u32>,
//
// inputViewportInPixelsX: f32,
// inputViewportInPixelsY: f32,
//
// inputSizeInPixelsX: f32,
// inputSizeInPixelsY: f32,
//
// outputSizeInPixelsX: f32,
// outputSizeInPixelsY: f32,
//
// inputOffsetInPixelsX: f32,
// inputOffsetInPixelsY: f32) {
//     FsrEasuCon(con0, con1, con2, con3, inputViewportInPixelsX, inputViewportInPixelsY, inputSizeInPixelsX, inputSizeInPixelsY, outputSizeInPixelsX, outputSizeInPixelsY);
//     con0[2] = floatBitsToUint(floatf32(float(0.5)) * inputViewportInPixelsX * ARcpF1(outputSizeInPixelsX) -f32(float(0.5)) + inputOffsetInPixelsX));
//     con0[3] = floatBitsToUint(floatf32(float(0.5)) * inputViewportInPixelsY * ARcpF1(outputSizeInPixelsY) -f32(float(0.5)) + inputOffsetInPixelsY));
// }

fn FsrEasuTapF(
    aC: ptr<function, vec3<f32>>,
    aW: ptr<function, f32>,
    off: vec2<f32>,
    dir: vec2<f32>,
    len: vec2<f32>,
    lob: f32,
    clp: f32,
    c: vec3<f32>
){
    var v: vec2<f32>;
    v.x=(off.x*( dir.x))+(off.y*dir.y);
    v.y=(off.x*(-dir.y))+(off.y*dir.x);

    v*=len;

    var d2=v.x*v.x+v.y*v.y;

    d2=min(d2,clp);

    var wB = (2.0/5.0)*d2 - 1.0;
    var wA=lob*d2 - 1.0;
    wB*=wB;
    wA*=wA;
    wB=(25.0/16.0)*wB-(25.0/16.0 - 1.0);
    let w=wB*wA;

    *aC+=c*w;*aW+=w;
}

fn FsrEasuSetF(
    dir: ptr<function, vec2<f32>>,
    len: ptr<function, f32>,
    pp: vec2<f32>,
    biS: bool, biT: bool, biU: bool, biV: bool,
    lA: f32, lB: f32, lC: f32, lD: f32, lE: f32
){
     var w = 0.0;
     if(biS) {w=(1.0-pp.x) * (1.0-pp.y);}
     if(biT) {w= pp.x * (1.0-pp.y);}
     if(biU) {w=(1.0-pp.x) * pp.y;}
     if(biV) {w= pp.x * pp.y ;}

     let dc=lD-lC;
     let cb=lC-lB;
     var lenX=max(abs(dc),abs(cb));
     lenX=APrxLoRcpF1(lenX);
     let dirX=lD-lB;
     (*dir).x+=dirX*w;
     lenX=ASatF1(abs(dirX)*lenX);
     lenX*=lenX;
     (*len)+=lenX*w;

     let ec=lE-lC;
     let ca=lC-lA;
     var lenY=max(abs(ec),abs(ca));
     lenY=APrxLoRcpF1(lenY);
     let dirY=lE-lA;
     (*dir).y+=dirY*w;
     lenY=ASatF1(abs(dirY)*lenY);
     lenY*=lenY;
     (*len)+=lenY*w;
}

fn FsrEasuF(
    pix: ptr<function, vec3<f32>>,
    ip: vec2<u32>,
    con0: vec4<u32>,
    con1: vec4<u32>,
    con2: vec4<u32>,
    con3: vec4<u32>
){
    var pp=vec2<f32>(ip)*bitcast<f32>(con0.xy)+bitcast<f32>(con0.zw);
    let fp=floor(pp);
    pp-=fp;

    let p0=fp*bitcast<f32>(con1.xy)+bitcast<f32>(con1.zw);

    let p1=p0+bitcast<f32>(con2.xy);
    let p2=p0+bitcast<f32>(con2.zw);
    let p3=p0+bitcast<f32>(con3.xy);
    let bczzR=FsrEasuRF(p0);
    let bczzG=FsrEasuGF(p0);
    let bczzB=FsrEasuBF(p0);
    let ijfeR=FsrEasuRF(p1);
    let ijfeG=FsrEasuGF(p1);
    let ijfeB=FsrEasuBF(p1);
    let klhgR=FsrEasuRF(p2);
    let klhgG=FsrEasuGF(p2);
    let klhgB=FsrEasuBF(p2);
    let zzonR=FsrEasuRF(p3);
    let zzonG=FsrEasuGF(p3);
    let zzonB=FsrEasuBF(p3);

    let bczzL=bczzB*vec4<f32>(0.5)+(bczzR*vec4<f32>(0.5)+bczzG);
    let ijfeL=ijfeB*vec4<f32>(0.5)+(ijfeR*vec4<f32>(0.5)+ijfeG);
    let klhgL=klhgB*vec4<f32>(0.5)+(klhgR*vec4<f32>(0.5)+klhgG);
    let zzonL=zzonB*vec4<f32>(0.5)+(zzonR*vec4<f32>(0.5)+zzonG);

    let bL=bczzL.x;
    let cL=bczzL.y;
    let iL=ijfeL.x;
    let jL=ijfeL.y;
    let fL=ijfeL.z;
    let eL=ijfeL.w;
    let kL=klhgL.x;
    let lL=klhgL.y;
    let hL=klhgL.z;
    let gL=klhgL.w;
    let oL=zzonL.z;
    let nL=zzonL.w;

    var dir=vec2<f32>(0.0);
    var len = 0.0;
    FsrEasuSetF(&dir,&len,pp,true, false,false,false,bL,eL,fL,gL,jL);
    FsrEasuSetF(&dir,&len,pp,false,true ,false,false,cL,fL,gL,hL,kL);
    FsrEasuSetF(&dir,&len,pp,false,false,true ,false,fL,iL,jL,kL,nL);
    FsrEasuSetF(&dir,&len,pp,false,false,false,true ,gL,jL,kL,lL,oL);

    var dir2=dir*dir;
    var dirR=dir2.x+dir2.y;
    let zro=dirR < (1.0/32768.0);
    dirR=APrxLoRsqF1(dirR);
    dirR = select(dirR, 1.0, zro);
    dir.x = select(dir.x, 1.0, zro);
    dir*=vec2<f32>(dirR);

    len=len * 0.5;
    len*=len;

    let stretch=(dir.x*dir.x+dir.y*dir.y)*APrxLoRcpF1(max(abs(dir.x),abs(dir.y)));

    let len2=vec2<f32>(1.0+(stretch - 1.0) * len, 1.0 + -0.5*len);

    let lob = 0.5 + ((1.0/4.0 - 0.04) - 0.5)*len;

    let clp=APrxLoRcpF1(lob);

    let min4=min(
        AMin3F3(vec3<f32>(ijfeR.z,ijfeG.z,ijfeB.z),vec3<f32>(klhgR.w,klhgG.w,klhgB.w),vec3<f32>(ijfeR.y,ijfeG.y,ijfeB.y)),
        vec3<f32>(klhgR.x,klhgG.x,klhgB.x)
    );
    let max4=max(
        AMax3F3(vec3<f32>(ijfeR.z,ijfeG.z,ijfeB.z),vec3<f32>(klhgR.w,klhgG.w,klhgB.w),vec3<f32>(ijfeR.y,ijfeG.y,ijfeB.y)),
        vec3<f32>(klhgR.x,klhgG.x,klhgB.x)
    );

    var aC=vec3<f32>(0.0);
    var aW = 0.0;
    FsrEasuTapF(&aC,&aW,vec2( 0.0,-1.0)-pp,dir,len2,lob,clp,vec3(bczzR.x,bczzG.x,bczzB.x));
    FsrEasuTapF(&aC,&aW,vec2( 1.0,-1.0)-pp,dir,len2,lob,clp,vec3(bczzR.y,bczzG.y,bczzB.y));
    FsrEasuTapF(&aC,&aW,vec2(-1.0, 1.0)-pp,dir,len2,lob,clp,vec3(ijfeR.x,ijfeG.x,ijfeB.x));
    FsrEasuTapF(&aC,&aW,vec2( 0.0, 1.0)-pp,dir,len2,lob,clp,vec3(ijfeR.y,ijfeG.y,ijfeB.y));
    FsrEasuTapF(&aC,&aW,vec2( 0.0, 0.0)-pp,dir,len2,lob,clp,vec3(ijfeR.z,ijfeG.z,ijfeB.z));
    FsrEasuTapF(&aC,&aW,vec2(-1.0, 0.0)-pp,dir,len2,lob,clp,vec3(ijfeR.w,ijfeG.w,ijfeB.w));
    FsrEasuTapF(&aC,&aW,vec2( 1.0, 1.0)-pp,dir,len2,lob,clp,vec3(klhgR.x,klhgG.x,klhgB.x));
    FsrEasuTapF(&aC,&aW,vec2( 2.0, 1.0)-pp,dir,len2,lob,clp,vec3(klhgR.y,klhgG.y,klhgB.y));
    FsrEasuTapF(&aC,&aW,vec2( 2.0, 0.0)-pp,dir,len2,lob,clp,vec3(klhgR.z,klhgG.z,klhgB.z));
    FsrEasuTapF(&aC,&aW,vec2( 1.0, 0.0)-pp,dir,len2,lob,clp,vec3(klhgR.w,klhgG.w,klhgB.w));
    FsrEasuTapF(&aC,&aW,vec2( 1.0, 2.0)-pp,dir,len2,lob,clp,vec3(zzonR.z,zzonG.z,zzonB.z));
    FsrEasuTapF(&aC,&aW,vec2( 0.0, 2.0)-pp,dir,len2,lob,clp,vec3(zzonR.w,zzonG.w,zzonB.w));

    (*pix)=min(max4,max(min4,aC*vec3<f32>(f32(ARcpF1(aW)))));
}
fn FsrRcasCon(
    con: ptr<function, vec4<u32>>,
    sharpness: f32,
) {
     let sharpness=exp2(-sharpness);
     let hSharp=vec2<f32>(sharpness,sharpness);
     (*con)[0]=bitcast<u32>(sharpness);
     (*con)[1]=pack2x16float(hSharp);
     (*con)[2]=0u;
     (*con)[3]=0u;
}

fn FsrRcasLoadF(p: vec2<i32>) -> vec4<f32> {
    return textureLoad(texture, p, 0);
}

fn FsrRcasInputF(r: ptr<function, f32>, g: ptr<function, f32>, b: ptr<function, f32>) {}

fn FsrRcasF(
    pixR: ptr<function, f32>,
    pixG: ptr<function, f32>,
    pixB: ptr<function, f32>,
    ip: vec2<u32>,
    con: vec4<u32>,
) {
     let sp=vec2<i32>(ip);
     let b=FsrRcasLoadF(sp+vec2<i32>( 0,-1)).rgb;
     let d=FsrRcasLoadF(sp+vec2<i32>(-1, 0)).rgb;

     let e=FsrRcasLoadF(sp).rgb;

     let f=FsrRcasLoadF(sp+vec2<i32>( 1, 0)).rgb;
     let h=FsrRcasLoadF(sp+vec2<i32>( 0, 1)).rgb;

     var bR=b.r;
     var bG=b.g;
     var bB=b.b;
     var dR=d.r;
     var dG=d.g;
     var dB=d.b;
     var eR=e.r;
     var eG=e.g;
     var eB=e.b;
     var fR=f.r;
     var fG=f.g;
     var fB=f.b;
     var hR=h.r;
     var hG=h.g;
     var hB=h.b;

     FsrRcasInputF(&bR,&bG,&bB);
     FsrRcasInputF(&dR,&dG,&dB);
     FsrRcasInputF(&eR,&eG,&eB);
     FsrRcasInputF(&fR,&fG,&fB);
     FsrRcasInputF(&hR,&hG,&hB);

     let bL=bB*0.5+(bR*0.5+bG);
     let dL=dB*0.5+(dR*0.5+dG);
     let eL=eB*0.5+(eR*0.5+eG);
     let fL=fB*0.5+(fR*0.5+fG);
     let hL=hB*0.5+(hR*0.5+hG);

     var nz = 0.25*bL+(0.25)*dL+(0.25)*fL+(0.25)*hL-eL;
     nz=ASatF1(abs(nz)*APrxMedRcpF1(AMax3F1(AMax3F1(bL,dL,eL),fL,hL)-AMin3F1(AMin3F1(bL,dL,eL),fL,hL)));
     nz=-0.5*nz+1.0;

     let mn4R=min(AMin3F1(bR,dR,fR),hR);
     let mn4G=min(AMin3F1(bG,dG,fG),hG);
     let mn4B=min(AMin3F1(bB,dB,fB),hB);
     let mx4R=max(AMax3F1(bR,dR,fR),hR);
     let mx4G=max(AMax3F1(bG,dG,fG),hG);
     let mx4B=max(AMax3F1(bB,dB,fB),hB);

     let peakC=vec2<f32>(1.0,-1.0*4.0);

     let hitMinR=min(mn4R,eR)*ARcpF1(4.0*mx4R);
     let hitMinG=min(mn4G,eG)*ARcpF1(4.0*mx4G);
     let hitMinB=min(mn4B,eB)*ARcpF1(4.0*mx4B);
     let hitMaxR=(peakC.x-max(mx4R,eR))*ARcpF1(4.0*mn4R+peakC.y);
     let hitMaxG=(peakC.x-max(mx4G,eG))*ARcpF1(4.0*mn4G+peakC.y);
     let hitMaxB=(peakC.x-max(mx4B,eB))*ARcpF1(4.0*mn4B+peakC.y);
     let lobeR=max(-hitMinR,hitMaxR);
     let lobeG=max(-hitMinG,hitMaxG);
     let lobeB=max(-hitMinB,hitMaxB);
     let lobe=max(-(0.25-(1.0/16.0)),min(AMax3F1(lobeR,lobeG,lobeB),0.0))*bitcast<f32>(con.x);

     let rcpL=APrxMedRcpF1(4.0*lobe+1.0);
     *pixR=(lobe*bR+lobe*dR+lobe*hR+lobe*fR+eR)*rcpL;
     *pixG=(lobe*bG+lobe*dG+lobe*hG+lobe*fG+eG)*rcpL;
     *pixB=(lobe*bB+lobe*dB+lobe*hB+lobe*fB+eB)*rcpL;
     return;
}

//fn FsrLfgaF(c: ptr<function, vec3<f32>, t: vec3<f32>, a: f32){c+=(t*vec3<f32>(a))*min(vec3<f32>(1.0)-c,c);}
//
// void FsrSrtmF(inout vec3 c){c*=vec3<f32>(float(ARcpF1(AMax3F1(c.r,c.g,c.b)f32(float(1.0)))));}
//
// void FsrSrtmInvF(inout vec3 c){c*=vec3<f32>(float(ARcpF1(maxf32(float(1.0/32768.0))f32(float(1.0))-AMax3F1(c.r,c.g,c.b)))));}
//
// float FsrTepdDitF(uvec2 p,uint f){
//     float xf32(float(p.x+f));
//     float yf32(float(p.y));
//
//     float af32(float((1.0+sqrt(5.0))/2.0));
//
//     float bf32(float(1.0/3.69));
//     x=x*a+(y*b);
//     return fract(x);}
//
// void FsrTepdC8F(inout vec3 c,float dit){
//     vec3 n=sqrt(c);
//     n=floor(n*vec3<f32>(float(255.0)))*vec3<f32>(float(1.0/255.0));
//     vec3 a=n*n;
//     vec3 b=n+vec3<f32>(float(1.0/255.0));b=b*b;
//
//     vec3 r=(c-b)*APrxMedRcpF3(a-b);
//
//     c=ASatF3(n+AGtZeroF3(vec3<f32>(float(dit))-r)*vec3<f32>(float(1.0/255.0)));}
//
// void FsrTepdC10F(inout vec3 c,float dit){
//     vec3 n=sqrt(c);
//     n=floor(n*vec3<f32>(float(1023.0)))*vec3<f32>(float(1.0/1023.0));
//     vec3 a=n*n;
//     vec3 b=n+vec3<f32>(float(1.0/1023.0));b=b*b;
//     vec3 r=(c-b)*APrxMedRcpF3(a-b);
//     c=ASatF3(n+AGtZeroF3(vec3<f32>(float(dit))-r)*vec3<f32>(float(1.0/1023.0)));}

@fragment
fn fragment(
    @location(0) _world_position: vec4<f32>,
    @location(1) _world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> @location(0) vec4<f32> {
    var pixel: vec3<f32>;
    let viewport_size = vec2<f32>(view.width, view.height);
    let scale_factor = upscale_settings.upscale_factor;
    var con0: vec4<u32>;
    var con1: vec4<u32>;
    var con2: vec4<u32>;
    var con3: vec4<u32>;

    FsrEasuCon(&con0, &con1, &con2, &con3,
     viewport_size.x * scale_factor, viewport_size.y * scale_factor,
     viewport_size.x, viewport_size.y,
     viewport_size.x, viewport_size.y
    );

    FsrEasuF(&pixel, vec2<u32>(floor(uv * viewport_size)), con0, con1, con2, con3);
    return vec4(pixel, 1.);
 }

@fragment
fn sharpen(
    @location(0) _world_position: vec4<f32>,
    @location(1) _world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>
) -> @location(0) vec4<f32> {
    let viewport_size = vec2<f32>(view.width, view.height);
    let sharpen = upscale_settings.sharpening_amount;
    var con: vec4<u32>;
    FsrRcasCon(&con, sharpen);

    var r: f32;
    var g: f32;
    var b: f32;
    let ip = vec2<u32>(floor(uv * viewport_size));

    FsrRcasF(&r, &g, &b, ip, con);

    return vec4<f32>(r, g, b, 1.);
}