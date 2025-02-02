import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import nmrglue as ng
import os
import tempfile
from mrsimulator import signal_processor as sp
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.relaxnmr.core import T1Functions
import matplotlib.pyplot as plt

class TestT1Functions(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp(prefix='t1_test_')
        self.t1_funcs = T1Functions(self.temp_dir)
        
        # Create mock data for testing
        self.mock_fid = (np.random.random(1024) + 1j * np.random.random(1024)).astype(np.complex128)
        self.mock_spectrum = (np.random.random(1024) + 1j * np.random.random(1024)).astype(np.complex128)
        self.mock_vd_list = np.array([0.001, 0.01, 0.1, 1.0, 2.0, 5.0])

    def tearDown(self):
        """Clean up after each test."""
        plt.close('all')
        if os.path.exists(self.temp_dir):
            for f in os.listdir(self.temp_dir):
                try:
                    os.remove(os.path.join(self.temp_dir, f))
                except (OSError, PermissionError):
                    pass
            try:
                os.rmdir(self.temp_dir)
            except (OSError, PermissionError):
                pass

    @patch('nmrglue.bruker.read')
    @patch('nmrglue.bruker.guess_udic')
    def test_read_and_convert_bruker_data(self, mock_guess_udic, mock_read):
        """Test reading and converting Bruker data."""
        # Create a properly structured udic with all required fields
        mock_udic = {
            "ndim": 2,
            0: {
                "encoding": "states",
                "sw": 50000,
                "obs": 400,
                "car": 100.0,
                "size": 1024,
                "label": "F2",
                "complex": True,
                "time": True,
                "freq": True
            },
            1: {
                "encoding": "states",
                "sw": 50000,
                "obs": 400,
                "car": 100.0,
                "size": 256,
                "label": "F1",
                "complex": True,
                "time": True,
                "freq": True
            }
        }
        mock_dic = {"ndim": 2}
        mock_data = np.zeros((256, 1024))
        
        mock_read.return_value = (mock_dic, mock_data)
        mock_guess_udic.return_value = mock_udic
        
        # Create the delay list file with proper path handling
        vdlist_path = os.path.join(self.temp_dir, "vdlist")
        np.savetxt(vdlist_path, self.mock_vd_list)
        
        # Patch the file path property to use temp_dir
        with patch.object(self.t1_funcs, 'file_path', self.temp_dir + os.path.sep):
            spectra, vd_list, csdm_ds = self.t1_funcs.read_and_convert_bruker_data(save_nmrpipe=False)
            
            self.assertIsInstance(vd_list, np.ndarray)
            self.assertEqual(len(vd_list), len(self.mock_vd_list))
            self.assertTrue(np.allclose(vd_list, self.mock_vd_list))
        self.assertIsInstance(vd_list, np.ndarray)
        self.assertEqual(len(vd_list), len(self.mock_vd_list))
                
    def test_zero_fill(self):
        """Test zero-filling functionality."""
        test_data = np.ones(100)
        new_len = 256
        filled_data = self.t1_funcs.zero_fill(test_data, new_len)
        self.assertEqual(len(filled_data), new_len)
        self.assertEqual(np.sum(filled_data[100:]), 0)

    def test_zero_order_phasing(self):
        """Test zero-order phase correction."""
        test_data = (np.ones(100) + 1j * np.zeros(100)).astype(np.complex128)
        phased_data = self.t1_funcs.zero_order_phasing(test_data, 90)
        self.assertTrue(np.allclose(np.real(phased_data), 0, atol=1e-10))
        self.assertTrue(np.allclose(np.imag(phased_data), 1, atol=1e-10))

    def test_first_order_phasing(self):
        """Test first-order phase correction."""
        test_data = (np.ones(100) + 1j * np.zeros(100)).astype(np.complex128)
        phased_data = self.t1_funcs.first_order_phasing(test_data, 45)
        self.assertTrue(np.all(np.abs(phased_data) - 1 < 1e-10))

    def test_process_spectrum(self):
        """Test spectrum processing functionality."""
        # Create a CSDM-like mock object
        mock_spectrum = MagicMock()
        mock_dimensions = [MagicMock()]
        mock_dimensions[0].coordinates = MagicMock()
        mock_dimensions[0].coordinates.value = np.arange(1024)
        mock_spectrum.dimensions = mock_dimensions
        mock_spectrum.shape = [1024]
        # Create the array data
        array_data = np.zeros(1024, dtype=np.complex128)
        mock_spectrum.__array__ = lambda *args: array_data
        
        with patch.object(sp.SignalProcessor, 'apply_operations') as mock_apply:
            # Return a similar mock object
            return_mock = MagicMock()
            return_mock.dimensions = mock_dimensions
            return_mock.shape = [1024]
            return_mock.__array__ = lambda *args: array_data
            mock_apply.return_value = return_mock
            
            result = self.t1_funcs.process_spectrum(
                mock_spectrum, 
                fwhm=10, 
                zero_fill_factor=2,
                ph0=0,
                ph1=0
            )
            mock_apply.assert_called_once()

    def test_integrate_spectrum_region(self):
        """Test spectrum integration functionality."""
        x = np.linspace(-10, 10, 1000)
        y = np.exp(-x**2)
        
        mock_spectrum = MagicMock()
        mock_dim = MagicMock()
        mock_dim.coordinates.value = x
        mock_spectrum.dimensions = [mock_dim]
        
        mock_var = MagicMock()
        mock_comp = MagicMock()
        mock_comp.real = y
        mock_var.components = [mock_comp]
        mock_spectrum.dependent_variables = [mock_var]
        
        trapz, simps, x_reg, y_reg, uncert = self.t1_funcs.integrate_spectrum_region(
            mock_spectrum, -2, 2
        )
        self.assertLess(abs(trapz - simps), 1e-3)

    def test_plot_spectra_and_zoomed_regions(self):
        """Test plotting functionality."""
        exp_spectra = []
        x_regions = []
        y_regions = []
        
        for _ in range(2):
            mock_spectrum = MagicMock()
            # Create proper numpy array for real attribute
            mock_spectrum.real = np.zeros(100)
            mock_spectrum.dependent_variables = [MagicMock()]
            mock_comp = MagicMock()
            mock_comp.max.return_value = 1.0
            mock_spectrum.dependent_variables[0].components = [mock_comp]
            # Add array-like behavior
            mock_spectrum.__array__ = lambda *args: mock_spectrum.real
            exp_spectra.append(mock_spectrum)
            x_regions.append(np.linspace(-5, 5, 100))
            y_regions.append(np.zeros(100))
        
        intensities = self.t1_funcs.plot_spectra_and_zoomed_regions(
            exp_spectra, x_regions, y_regions, -5, 5
        )
        self.assertEqual(len(intensities), 2)
        
    def test_mono_satrec_func(self):
        """Test mono-exponential saturation recovery function."""
        t = np.linspace(0, 20, 1000)
        M0, T1, A, B = 1.0, 2.0, 1.0, 0.0
        result = self.t1_funcs.mono_satrec_func(t, M0, T1, A, B)
        
        # Test dimensions
        self.assertEqual(len(result), len(t))
        
        # Test initial and final values
        self.assertAlmostEqual(result[0], B, places=3)  # Initial value
        self.assertAlmostEqual(result[-1], A*M0 + B, places=3)  # Final value

    def test_di_satrec_func(self):
        """Test bi-exponential saturation recovery function."""
        t = np.linspace(0, 20, 1000)
        result = self.t1_funcs.di_satrec_func(t, M0=0.7, T1=2.0, A=1.0, M1=0.3, T2=0.5)
        self.assertEqual(len(result), len(t))
        self.assertTrue(np.all(result >= 0))

    def test_tri_satrec_func(self):
        """Test tri-exponential saturation recovery function."""
        t = np.linspace(0, 20, 1000)
        result = self.t1_funcs.tri_satrec_func(
            t, M0=0.5, T1=2.0, A=1.0, M1=0.3, T2=0.5, M2=0.2, T3=0.1
        )
        self.assertEqual(len(result), len(t))
        self.assertTrue(np.all(result >= 0))

    def test_stretch_t1_exponential(self):
        """Test stretched exponential function."""
        t = np.linspace(0, 10, 100)
        
        # Test normal behavior
        result = self.t1_funcs.stretch_t1_exponential(t, T1_star=2.0, c=0.5)
        self.assertEqual(len(result), len(t))
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))
        
        # Test with c=1 (should reduce to normal exponential)
        result_normal = self.t1_funcs.stretch_t1_exponential(t, T1_star=2.0, c=1.0)
        result_stretched = self.t1_funcs.stretch_t1_exponential(t, T1_star=2.0, c=0.5)
        self.assertTrue(np.any(np.not_equal(result_normal, result_stretched)))

    def test_mono_expdec(self):
        """Test mono-exponential decay function."""
        t = np.linspace(0, 20, 100)  # Extended time range
        
        # Test normal behavior
        result = self.t1_funcs.mono_expdec(t, T1=2.0, A=1.0, B=1.0, C=0.0)
        self.assertEqual(len(result), len(t))
        
        # Test decay to baseline with relaxed tolerance
        self.assertAlmostEqual(result[-1], 0.0, places=1)
        
        # Test with offset
        result = self.t1_funcs.mono_expdec(t, T1=2.0, A=1.0, B=1.0, C=0.5)
        self.assertAlmostEqual(result[-1], 0.5, places=1)

    def test_di_expdec(self):
        """Test bi-exponential decay function."""
        t = np.linspace(0, 20, 100)
        result = self.t1_funcs.di_expdec(t, T1=2.0, T2=0.5, A=1.0, C=0.7, D=0.3)
        self.assertEqual(len(result), len(t))
        self.assertTrue(result[-1] < 0.1)

    def test_tri_expdec(self):
        """Test tri-exponential decay function."""
        t = np.linspace(0, 20, 100)
        result = self.t1_funcs.tri_expdec(
            t, T1=2.0, T2=0.5, T3=0.1, A=1.0, C=0.5, D=0.3, E=0.2
        )
        self.assertEqual(len(result), len(t))
        self.assertTrue(result[-1] < 0.1)

    def test_tri_expdec(self):
        """Test tri-exponential decay function."""
        t = np.linspace(0, 20, 100)
        result = self.t1_funcs.tri_expdec(
            t, T1=2.0, T2=0.5, T3=0.1, A=1.0, C=0.5, D=0.3, E=0.2
        )
        self.assertEqual(len(result), len(t))
        self.assertTrue(result[-1] < 0.1)
        
if __name__ == '__main__':
    
    cov = coverage.Coverage()
    cov.start()
    
    unittest.main(verbosity=2)
    
    cov.stop()
    
    cov.save()
    
    cov.html_report(directory='coverage_html')