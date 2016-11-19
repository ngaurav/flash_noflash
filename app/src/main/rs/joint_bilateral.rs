#pragma version(1)
#pragma rs java_package_name(net.sourceforge.opencamera)
#pragma rs_fp_relaxed

// size of Gauss matrix (bilateral filter - pixel position distance)
#define GM_SIZE 5
// size of Gauss matrix (noice reduction)
#define GSM_SIZE 3
// sigma of gauss function (bilateral filter - pixel intensity distance)
#define GAUSS_SIGMA 0.1f // for intensity values in float3 image space

// pre-evaluated multiplier for transforming byte RGBa values [0;255] to float RGB [0;1]
const float normU8 = 1.f/255.f;
// radiuses of matrices used
const uint gmBorder = (GM_SIZE - 1) / 2;
const uint gsmBorder = (GSM_SIZE - 1) / 2;

// filter output resolution parameters
uint sImageWidth;
uint sImageHeight;

// pre-evaluated multiplier in gauss function exponent (bilateral filter)
// they are evaluated in "init" function
float gaussExpMult;
float gaussMult;

// Gauss matrix used on camera input image to reduce noice by smoothing
static float gaussSmoothMatrix[GSM_SIZE][GSM_SIZE] = {
    {0.109021f,	0.112141f,	0.109021f},
    {0.112141f,	0.11535f,	0.112141f},
    {0.109021f,	0.112141f,	0.109021f}
};

// Gauss matrix used in bilateral filter (pixel position distance)
static float gaussMatrix[GM_SIZE][GM_SIZE] = {
    {0.010534f,	0.02453f,	0.032508f,	0.02453f,	0.010534f},
    {0.02453f,	0.05712f,	0.075698f,	0.05712f,	0.02453f},
    {0.032508f,	0.075698f,	0.100318f,	0.075698f,	0.032508f},
    {0.02453f,	0.05712f,	0.075698f,	0.05712f,	0.02453f},
    {0.010534f,	0.02453f,	0.032508f,	0.02453f,	0.010534f}
};

// buffer containing input image buffer for kernels
rs_allocation noFlashBuffer;
rs_allocation flashBuffer;

inline static float3 zero_float3() {
    float3 c;
    c.r = 0.f;
    c.g = 0.f;
    c.b = 0.f;
    return c;
}

// Evaluates value of a gauss function (normal distribution with pre-evaluated exponent multiplier)
inline static float gauss(float x) {
    return gaussMult * native_exp(gaussExpMult * x * x);
}

// Evaluates intensity of an RGB pixel = lenght of the vector
inline static float intensity(float3 rgb) {
    return fast_length(rgb);
}

// Evaluates intensity difference between two RGB pixels
// Applies logarithm and division to reduce "halo" effect around edges
inline static float intensityDifference(float3 x, float3 y) {
    return native_log((intensity(x) + 0.01f) / (intensity(y) + 0.01f));
}

void init() {
    gaussExpMult = - 1.f / (2.f * GAUSS_SIGMA * GAUSS_SIGMA);
    gaussMult = 1.f / sqrt(2.f * M_PI * GAUSS_SIGMA * GAUSS_SIGMA);
}

void __attribute__((kernel)) initFlashBuffer(uchar4 in, uint32_t x, uint32_t y) {
    float3 fRGB;
    fRGB.r = (float) normU8 * in.r;
    fRGB.g = (float) normU8 * in.g;
    fRGB.b = (float) normU8 * in.b;
    rsSetElementAt_float3(flashBuffer, fRGB, x, y);
}

void __attribute__((kernel)) initNoFlashBuffer(uchar4 in, uint32_t x, uint32_t y) {
    float3 fRGB;
    fRGB.r = normU8 * in.r;
    fRGB.g = normU8 * in.g;
    fRGB.b = normU8 * in.b;
    rsSetElementAt_float3(noFlashBuffer, fRGB, x, y);
}

// Applies bilateral filter on an image
uchar4 __attribute__((kernel)) bilateral(uchar4 in, uint32_t x, uint32_t y) {
    float3 result = zero_float3();
    float W = 0.f;

    for (uint i = 0; i < GM_SIZE; ++i) {
        for (uint j = 0; j < GM_SIZE; ++j) {
            float gm = gaussMatrix[i][j];
            float3 noflash_pixel = rsGetElementAt_float3(noFlashBuffer, clamp(x + i - gmBorder, 0u, sImageWidth - 1u), clamp(y + j - gmBorder, 0u, sImageHeight - 1u));
            float3 pixel = rsGetElementAt_float3(flashBuffer, clamp(x + i - gmBorder, 0u, sImageWidth - 1u), clamp(y + j - gmBorder, 0u, sImageHeight - 1u));
            float3 origin = rsGetElementAt_float3(flashBuffer, x, y);
            float ev = gm * gauss(intensityDifference(pixel, origin));
            W += ev;
            result += ev * noflash_pixel;
        }
    }

    result = rsGetElementAt_float3(flashBuffer, x, y);;
    uchar4 out;
    out.r = (uchar)(255 * result.r);
    out.g = (uchar)(255 * result.g);
    out.b = (uchar)(255 * result.b);
    out.a = 255;
    return out;
}
