import nmrglue as ng
import numpy as np
from mrsimulator import signal_processor as sp
from scipy.integrate import simpson, trapezoid
import matplotlib.pyplot as plt

class T1Functions:
    
    """
    A comprehensive class for processing and analyzing relaxation NMR data from Bruker format.
    This class provides tools for data conversion, processing, visualization, and fitting of relaxation data.
    It supports various fitting models including mono-, bi-, and tri-exponential functions, as well as
    stretched exponentials for complex relaxation behavior analysis.
    """
     
    def __init__(self, file_path):
        
        """
        Initialize T1Functions with a path to Bruker NMR data.
        
        Args:
            file_path (str): 
            Path to the Bruker NMR data directory containing the FID and acquisition 
            parameters.
            The directory should contain standard Bruker files (fid, acqu, acqus, etc.)
        """
        self.file_path = file_path

    def read_and_convert_bruker_data(self, save_nmrpipe=True):
        
        """
        Read and convert Bruker NMR data to NMRPipe and CSDM formats. 
        This function handles the complex
        process of importing raw Bruker data, 
        interpreting the acquisition parameters, and converting
        the data into formats suitable for further analysis. 
        It automatically detects and loads the
        variable delay list (vdlist, vplist, or vclist) used in the T1 experiment.
        
        Args:
            save_nmrpipe (bool): Whether to save the converted NMRPipe data to disk. 
            NMRPipe format is widely used in the NMR community and 
            can be processed with various
                                third-party tools.
            
        Returns:
            tuple: A tuple containing three elements:
                - list of 1D spectra: Each element 
                    is a single FID from the pseudo-2D dataset
                - variable delay list: The time delays used in the T1 experiment
                - CSDM dataset: The complete dataset in CSDM format for advanced processing
            
        Raises:
            FileNotFoundError: If no variable delay list file 
                                (vdlist, vplist, or vclist) is found
                                    in the Bruker data directory
        """
        # Read Bruker data
        dic, data = ng.bruker.read(self.file_path)
        u = ng.bruker.guess_udic(dic, data)

        # Create the converter object and initialize with Bruker data
        C = ng.convert.converter()
        C.from_bruker(dic, data, u)

        # Optionally save NMRPipe formatted data
        if save_nmrpipe:
            ng.pipe.write(self.file_path + "2d_pipe.fid", *C.to_pipe(), overwrite=True)

        # Convert to CSDM format
        csdm_ds = C.to_csdm()
        dim1, dim2 = csdm_ds.shape

        # Extract 1D spectra from the 2D dataset
        spectra_1d = [csdm_ds[:, i] for i in range(dim2)]
        
        possible_files = ["vdlist", "vplist", "vclist"]
        
        for filename in possible_files:
            
            try:
                vd_list = np.loadtxt(self.file_path + filename)
                break
            except OSError:
                
                if filename == possible_files[-1]:
                    raise FileNotFoundError("No vdlist, vplist, or vclist file found.")
                continue
            
        return spectra_1d, vd_list, csdm_ds

    def zero_fill(self, data, new_len):
        """
        Zero-fill NMR data to extend its length, 
        improving spectral resolution in the frequency domain.
        Zero-filling is a crucial preprocessing step that increases the 
        digital resolution of the spectrum
        by extending the FID with zeros. 
        This does not add any new information but provides interpolation
        in the frequency domain, resulting in smoother spectral lines.
        
        Args:
            data (ndarray): Input NMR data array, typically a time-domain FID
            new_len (int): Desired length after zero-filling. 
            Should be greater than the original
                          data length, typically a power of 2 for efficient FFT processing
            
        Returns:
            ndarray: Zero-filled data array with length new_len.
            If new_len is less than or equal
                    to the current length, returns the original data unchanged
        """
        current_len = data.shape[0]
        if new_len <= current_len:
            return data
        zeros_to_add = new_len - current_len
        return np.pad(data, (0, zeros_to_add), 'constant')

    def zero_order_phasing(self, data, ph0):
        """
        Apply zero-order phase correction to NMR data.
        Zero-order phasing applies a constant phase
        adjustment across the entire spectrum,
        correcting for the receiver phase offset during
        signal acquisition.
        This is essential for obtaining pure absorption mode spectra and is
        typically the first step in phase correction.
        
        Args:
            data (ndarray): Complex input NMR data array in either time or frequency domain
            ph0 (float): Phase angle in degrees. The phase correction is applied uniformly
                        across the entire spectrum. Typical values range from -180° to +180°
            
        Returns:
            ndarray: Phase-corrected complex data array.
            The correction is applied by multiplying
                    the data by exp(i*φ), where φ is the phase angle in radians
        """
        phase = np.deg2rad(ph0)
        
        phased_data = data * np.exp(1j * phase)
        
        return phased_data

    def first_order_phasing(self, data, ph1):
        """
        Apply first-order phase correction to NMR data.
        First-order phasing applies a frequency-dependent
        phase correction that varies linearly across the spectrum.
        This corrects for delays between
        excitation and detection, digital filtering effects,
        and other instrumental factors that can
        cause frequency-dependent phase errors.
        
        Args:
            data (ndarray): Complex input NMR data array, typically in the frequency domain
            ph1 (float): First-order phase correction factor. 
            This determines the slope of the
                        phase correction across the spectrum. 
                        The actual phase correction at each
                        point is ph1 * frequency
            
        Returns:
            ndarray: Phase-corrected complex data array with
            frequency-dependent phase adjustment
        """
        n = data.shape[0]
        ppm = np.linspace(-n//2, n//2, n)
        
        phase = np.deg2rad(ph1*ppm)
        
        phased_data = data * np.exp(1j * phase)
        
        return phased_data

    def process_spectrum(self, spectrum, fwhm, zero_fill_factor, ph0, ph1):
        """
        Process NMR spectrum with a comprehensive set of
        standard NMR data processing steps.
        This function applies, in order:
        1. Gaussian apodization for line broadening and S/N improvement
        2. Zero-filling for increased digital resolution
        3. Fourier transformation to convert from time to frequency domain
        4. Phase corrections (both zero- and first-order)
        5. Conversion to ppm scale for chemical shift referencing
        
        Args:
            spectrum (ndarray): Input time-domain NMR spectrum (FID)
            fwhm (float): Full width at half maximum for Gaussian apodization in Hz.
                         Controls the trade-off between resolution and signal-to-noise
            zero_fill_factor (int): Factor for zero filling, typically 2-4 for moderate
                                   resolution enhancement
            ph0 (float): Zero-order phase correction in degrees
            ph1 (float): First-order phase correction factor
            
        Returns:
            ndarray: Fully processed frequency-domain spectrum referenced to ppm scale
        """
        # Apply line broadening and Fourier transform
        
        ft = sp.SignalProcessor(operations=[sp.apodization.Gaussian(FWHM=fwhm), sp.FFT()])
        
        # Apply zero filling
        spectrum = self.zero_fill(spectrum, zero_fill_factor * spectrum.shape[0])
        
        # Apply first order phasing
        exp_spectrum = self.zero_order_phasing(spectrum, ph0)
        
        # Apply operations from the signal processor
        exp_spectrum = ft.apply_operations(dataset=exp_spectrum)
        
        # Apply second order phasing
        exp_spectrum = self.first_order_phasing(exp_spectrum, ph1)
        
        # Convert to ppm
        exp_spectrum.dimensions[0].to("ppm", "nmr_frequency_ratio")
        
        return exp_spectrum

    def integrate_spectrum_region(self, exp_spectrum, ppm_start, ppm_end):
        """
        Calculate integrated intensity of a spectral region using multiple
        numerical integration
        methods. 
        This function provides robust integration by comparing different numerical
        integration techniques
        (trapezoid and Simpson's rules) and estimating the uncertainty
        in the integration.
        This is particularly useful for quantitative NMR analysis and
        relaxation measurements.
        
        Args:
            exp_spectrum (ndarray): Input frequency-domain NMR spectrum
            ppm_start (float): Starting chemical shift in ppm for the integration region
            ppm_end (float): Ending chemical shift in ppm for the integration region
            
        Returns:
            tuple: A comprehensive set of integration results:
                - trapezoid integration value
                - Simpson's rule integration value
                - x coordinates of the integrated region
                - y coordinates of the integrated region
                - estimated integration uncertainty (difference between methods)
        """
        # Convert the ppm range to indices
        ppm_scale = exp_spectrum.dimensions[0].coordinates.value
        
        # Create a mask for the region of interest
        region_mask = (ppm_scale >= ppm_start) & (ppm_scale <= ppm_end)
        
        # Extract the region of interest
        x_region = ppm_scale[region_mask]
        y_real = exp_spectrum.dependent_variables[0].components[0].real
        y_region = y_real[region_mask]
        
        # Calculate integrated intensity using trapezoid and Simpson's rule methods
        integrated_intensity_trapz = trapezoid(y=y_region, x=x_region)
        integrated_intensity_simps = simpson(y=y_region, x=x_region)
        # integrated_intensity_romb = romb(y=y_region, dx=x_region[1]-x_region[0])
        integrated_uncertainty = abs(integrated_intensity_trapz - integrated_intensity_simps)
        
        return integrated_intensity_trapz, integrated_intensity_simps, x_region, y_region, integrated_uncertainty

    def plot_spectra_and_zoomed_regions(self, exp_spectra, x_regions, y_regions, xlim1, xlim2):
        """
        Create publication-quality plots of
        NMR spectra with both full view and zoomed regions.
        This visualization function creates a two-panel figure showing:
        1. The full spectrum for context
        2. A zoomed view of specific regions of interest
        
        The zoomed regions can be highlighted for emphasis, making it easy to focus on
        specific spectral features while maintaining the context of the full spectrum.
        
        Args:
            exp_spectra (list): List of processed NMR spectra to display
            x_regions (list): X coordinates for each region to be highlighted
            y_regions (list): Y coordinates for each highlighted region
            xlim1 (float): Lower chemical shift limit for the zoomed view (in ppm)
            xlim2 (float): Upper chemical shift limit for the zoomed view (in ppm)
            
        Returns:
            list: Maximum intensities from each spectrum, useful for normalization
                 and comparison between spectra
        """
        intensities = []
        
        
        for i, exp_spectrum in enumerate(exp_spectra):
            # if i == 0:
            fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), subplot_kw={"projection": "csdm"})
            
            ax[0].plot(exp_spectrum.real)
            ax[0].set_title(f"Full Spectrum {i+1}")
            ax[0].invert_xaxis()
            ax[1].plot(exp_spectrum.real, label="real")
            ax[1].fill_between(x_regions[i], y_regions[i], color='red', alpha=0.5)
            ax[1].set_title(f"Zoomed Spectrum {i+1}")
            ax[1].invert_xaxis()
            ax[1].set_xlim(xlim1, xlim2) #make this modular by passing x_lim as a parameter
            
            intensity = np.abs(exp_spectrum.dependent_variables[0].components[0].max())
            intensities.append(intensity)
        
        plt.tight_layout()
        plt.legend()
        plt.show()
        plt.close()
        
        return intensities


    def mono_satrec_func(self, t, M0, T1, A, B):
        """
        Single-component saturation recovery function for T1 fitting.
        This model describes
        the simplest case of longitudinal relaxation where a single population of spins
        returns to equilibrium following an exponential recovery. It follows the equation:
        M(t) = A*M0*(1 - exp(-t/T1)) + B
        
        This model is appropriate for systems with a single well-defined relaxation process,
        such as pure liquids or mobile species in solution.
        
        Args:
            t (ndarray): Time points of the relaxation curve
            M0 (float): Equilibrium magnetization, representing the fully relaxed signal
            T1 (float): Spin-lattice relaxation time constant
            A (float): Scaling factor to account for experimental conditions
            B (float): Baseline offset to account for instrumental effects
            
        Returns:
            ndarray: Calculated magnetization values at each time point
        """
        return A*M0 * (1 - np.exp(-t / T1)) + B
    
    def di_satrec_func(self, t, M0, T1, A, M1, T2):
        """
        Two-component saturation recovery function for T1 fitting. 
        This model describes systems
        with two distinct populations of spins, each with its own relaxation time constant.
        The function follows the equation:
        M(t) = A*[M0*(1 - exp(-t/T1)) + M1*(1 - exp(-t/T2))]
        
        This model is useful for heterogeneous systems, such as:
        - Different chemical environments in solids
        - Multiple phases in materials
        - Systems with distinct mobility regions
        
        Args:
            t (ndarray): Time points of the relaxation curve
            M0, M1 (float): Equilibrium magnetizations for each component
            T1, T2 (float): Relaxation time constants for each component
            A (float): Overall scaling factor
            
        Returns:
            ndarray: Combined magnetization values from both components
        """
        return A* ( (M0 * (1 - np.exp(-t / T1))) + (M1 * (1 - np.exp(-t / T2))) )

    def tri_satrec_func(self, t, M0, T1, A, M1, T2, M2, T3):
        """
        Three-component saturation recovery function for T1 fitting. 
        This model handles complex
        systems with three distinct relaxation processes. 
        The function follows the equation:
        M(t) = A*[M0*(1 - exp(-t/T1)) + M1*(1 - exp(-t/T2)) + M2*(1 - exp(-t/T3)^2)]
        
        This sophisticated model is applicable to:
        - Complex heterogeneous materials
        - Multi-phase systems
        - Materials with distinct domains of different mobilities
        - Systems with both surface and bulk relaxation processes
        
        Args:
            t (ndarray): Time points of the relaxation curve
            M0, M1, M2 (float): Equilibrium magnetizations for each component
            T1, T2, T3 (float): Relaxation time constants for each component
            A (float): Overall scaling factor
            
        Returns:
            ndarray: Combined magnetization values from all three components
        """
        return A* ( (M0 * (1 - np.exp(-t / T1))) + (M1 * (1 - np.exp(-t / T2))) + 
                   (M2 * (1 - np.exp(-t / T3))**2) )

    def stretch_t1_exponential(self, t, T1_star, c):
        """
        Stretched exponential function for non-standard T1 relaxation behavior. 
        This model
        accounts for systems with a continuous distribution of relaxation times,
        following
        the Kohlrausch-Williams-Watts (KWW) function:
        M(t) = 1 - exp(-(t/T1_star)^c)
        
        This model is particularly useful for:
        - Disordered systems
        - Glasses and polymers
        - Systems with complex relaxation dynamics
        - Materials with a distribution of correlation times
        
        The stretching exponent c (0 < c ≤ 1) indicates the degree of deviation from
        simple exponential behavior, with c = 1 recovering the standard exponential case.
        
        Args:
            t (ndarray): Time points of the relaxation curve
            T1_star (float): Characteristic relaxation time
            c (float): Stretching exponent, typically between 0 and 1
            
        Returns:
            ndarray: Stretched exponential relaxation curve values
        """
        return (1 - np.exp(-(t / T1_star)**c))
    
    
    def mono_expdec(self, t, T1, A, B, C):
        """
        Single-component exponential decay function for T1 relaxation analysis. This model
        describes the decay of magnetization following inversion or saturation, following
        the equation:
        M(t) = A*[C*exp(-t/T1) + C]
        
        This is the simplest decay model, appropriate for:
        - Homogeneous samples
        - Simple liquids
        - Systems with a single relaxation environment
        
        Args:
            t (ndarray): Time points of the decay curve
            T1 (float): Relaxation time constant
            A (float): Overall amplitude scaling factor
            C (float): Equilibrium offset
            
        Returns:
            ndarray: Exponential decay values at each time point
        """
        
        return A * ((B)*np.exp(-t/T1)) + C
    
    def di_expdec(self, t, T1, T2, A, C, D):
        """
        Two-component exponential decay function for complex T1 relaxation analysis.
        This model
        combines two independent decay processes, following the equation:
        M(t) = A*[C*exp(-t/T1) + D*exp(-t/T2)]
        
        Useful for analyzing:
        - Two-phase systems
        - Materials with distinct mobility regions
        - Systems with both surface and bulk relaxation
        - Heterogeneous materials with two distinct environments
        
        Args:
            t (ndarray): Time points of the decay curve
            T1, T2 (float): Relaxation time constants for each component
            A (float): Overall amplitude scaling factor
            C, D (float): Individual component scaling factors
            
        Returns:
            ndarray: Combined decay values from both components
        """
        
        return A * (((C)*np.exp(-t/T1)) + ((D)*np.exp(-t/T2)))
    
    def tri_expdec(self, t, T1, T2, T3, A, C, D, E):
        """
        Three-component exponential decay function for complex T1 relaxation analysis.
        This model
        describes systems with three distinct relaxation processes, following the equation:
        M(t) = A*[C*exp(-t/T1) + D*exp(-t/T2) + E*exp(-t/T3)]
        
        This sophisticated model is suitable for:
        - Highly heterogeneous materials
        - Multi-phase systems
        - Complex biological samples
        - Materials with multiple distinct chemical environments
        - Systems with multiple mobility regions
        
        Args:
            t (ndarray): Time points of the decay curve
            T1, T2, T3 (float): Relaxation time constants for each component
            A (float): Overall amplitude scaling factor
            C, D, E (float): Individual component scaling factors
            
        Returns:
            ndarray: Combined decay values from all three components
        """
        
        return A * (((C)*np.exp(-t/T1)) + ((D)*np.exp(-t/T2)) + ((E)*np.exp(-t/T3)))
    
    def stretch_expdec(self, t, T1, A, B):
    
        """
        Stretched exponential decay function for non-standard relaxation behavior. This model
        accounts for systems with a continuous distribution of relaxation times, following
        a modified Kohlrausch-Williams-Watts (KWW) function:
        M(t) = A*exp[-(t/T1)^B]
        
        This model is particularly valuable for:
        - Amorphous materials
        - Polymers and glasses
        - Systems with heterogeneous dynamics
        - Materials with complex structural organizations
        - Systems with correlated relaxation processes
        
        The stretching exponent B characterizes the distribution width of relaxation times,
        with B = 1 corresponding to simple exponential decay and B < 1 indicating
        increasing heterogeneity in the relaxation process.
        
        Args:
            t (ndarray): Time points of the decay curve
            T1 (float): Characteristic relaxation time constant
            A (float): Overall amplitude scaling factor
            B (float): Stretching exponent, typically between 0 and 1
            
        Returns:
            ndarray: Stretched exponential decay values at each time point,
                    representing the complex relaxation behavior of the system
        """
        
        return A * np.exp((-t/T1)**B)
