"""
Module for the LJT/YFOSC instrument

.. include:: ../include/links.rst
"""
from IPython.terminal.embed import embed
# from pkg_resources import resource_filename

import numpy as np

from astropy.time import Time

from pypeit import msgs
from pypeit import telescopes
from pypeit import io
from pypeit.core import framematch
from pypeit.spectrographs import spectrograph
from pypeit.core import parse
from pypeit.images import detector_container

class LJTYFOSCSpectrograph(spectrograph.Spectrograph):
    """
    Child to handle LJT/YFOSC spectrograph
    """
    ndet = 1
    name = 'ljt_yfosc'
    telescope = telescopes.LijiangTelescopePar()
    camera = 'YFOSC'
    url = 'https://ynao.cas.cn/kyzb/202306/t20230617_6779436.html'
    header_name = 'yf01'
    pypeline = 'MultiSlit'
    supported = True
    # ech_fixed_format = True
    comment = 'For use with the standard horizontal slits only. Grism 3'
    allowed_extensions = ['.fits.gz','.fz']

    def get_detector_par(self, det, hdu=None):
        """
        Return metadata for the selected detector.

        .. warning::

            Many of the necessary detector parameters are read from the file
            header, meaning the ``hdu`` argument is effectively **required** for
            NOT/ALFOSC.  The optional use of ``hdu`` is only viable for
            automatically generated documentation.

        Args:
            det (:obj:`int`):
                1-indexed detector number.
            hdu (`astropy.io.fits.HDUList`_, optional):
                The open fits file with the raw image of interest.  If not
                provided, frame-dependent parameters are set to a default.

        Returns:
            :class:`~pypeit.images.detector_container.DetectorContainer`:
            Object with the detector metadata.
        """

        # https://pypeit.readthedocs.io/en/release/api/pypeit.images.detector_container.html#module-pypeit.images.detector_container

        if hdu is None:
            binning = '1,1'
            gain = None
            ronoise = None
        else:
            # binning = np.atleast_1d(hdu[1].header['CCDSUM'])
            binning = self.get_meta_value(self.get_headarr(hdu), 'binning')
            gain = np.atleast_1d(hdu[1].header['GAIN'])  # [electrons/count] Pixel gain
            ronoise = np.atleast_1d(hdu[1].header['RDNOISE'])  # [electrons/pixel] Read noise
            darkcurr = hdu[1].header['DARKCURR']  # [electrons/pixel/s @ 200K] Dark current
            saturation = hdu[1].header['SATURATE']  # [ADU] Saturation level
            maxlin = hdu[1].header['MAXLIN']  # [ADU] Non-linearity level
            pixscale = hdu[1].header['PIXSCALE']  # [arcsec/pixel] Nominal pixel scale on sky
            datasec = np.atleast_1d('[2100:4100,500:1600]')
            #np.atleast_1d(hdu[1].header['DATASEC']) [1:2148,1:4612]
            overscan = None
            #np.atleast_1d(hdu[1].header['BIASSEC']) [2100:2148,50:4612]

        # Detector 1
        detector_dict = dict(
            binning         = binning,#Binning in PypeIt orientation (not the original)
            darkcurr        = darkcurr*3600.,#Dark current (e-/pixel/hour)
            dataext         = 1,#Index of fits extension containing data
            datasec         = datasec,#Either the data sections or the header keyword where the valid data sections can be obtained, one per amplifier. If defined explicitly should be in FITS format (e.g., [1:2048,10:4096]).
            det             = det,#PypeIt designation for detector number (1-based).
            gain            = gain,#Inverse gain (e-/ADU). A list should be provided if a detector contains more than one amplifier.
            mincounts       = -1e10,#Counts (e-) in a pixel below this value will be ignored as being unphysical.
            nonlinear       = maxlin/saturation,#Percentage of detector range which is linear (i.e. everything above nonlinear*saturation will be flagged as saturated)
            numamplifiers   = 1,
            oscansec        = overscan,#Either the overscan section or the header keyword where the valid data sections can be obtained, one per amplifier. If defined explicitly should be in FITS format (e.g., [1:2048,10:4096]).
            platescale      = pixscale,#arcsec per pixel in the spatial dimension for an unbinned pixel
            ronoise         = ronoise,#Read-out noise (e-). A list should be provided if a detector contains more than one amplifier. If any element of this list is <=0, the readout noise will be determined from the overscan regions defined by oscansec.
            saturation      = saturation,#The detector saturation level in ADU/DN
            spatflip        = False,#If this is True then the spatial dimension will be flipped. PypeIt expects echelle orders to increase with increasing pixel number. I.e., setting spatflip=True can reorder images so that blue orders appear on the left and red orders on the right.
            specaxis        = 0,#Spectra are dispersed along this axis. Allowed values are 0 (first dimension for a numpy array shape) or 1 (second dimension for numpy array shape).
            specflip        = False,#If this is True then the dispersion dimension (specified by the specaxis) will be flipped. PypeIt expects wavelengths to increase with increasing pixel number. If this is not the case for this instrument, set specflip to True.
            xgap            = 0.,#Gap between the square detector pixels (expressed as a fraction of the x pixel size – x is predominantly the spatial axis)
            ygap            = 0.,#Gap between the square detector pixels (expressed as a fraction of the y pixel size – y is predominantly the spectral axis)
            ysize           = 1.,#The size of a pixel in the y-direction as a multiple of the x pixel size (i.e. xsize = 1.0 – x is predominantly the dispersion axis)
        )

        # Return
        return detector_container.DetectorContainer(**detector_dict)

    @classmethod
    def default_pypeit_par(cls):
        """
        Return the default parameters to use for this instrument.

        Returns:
            :class:`~pypeit.par.pypeitpar.PypeItPar`: Parameters required by
            all of PypeIt methods.
        """
        # https://pypeit.readthedocs.io/en/release/pypeit_par.html
        par = super().default_pypeit_par()
        
        # Set the default exposure time ranges for the frame typing
        par['calibrations']['darkframe']['exprng'] = [999999, None] # No dark frames
        par['calibrations']['pinholeframe']['exprng'] = [999999, None] # No pinhole frames
        par['calibrations']['biasframe']['exprng'] = [None, 1] # bias frame
        par['calibrations']['pixelflatframe']['exprng'] = [10, 30]  # flat frame
        par['calibrations']['standardframe']['exprng'] = [None, 120]
        par['calibrations']['arcframe']['exprng'] = [None, 30]  # Arc exposures
        par['scienceframe']['exprng'] = [120, None] # Science exposures
        par['scienceframe']['process']['spat_flexure_correct'] = True

        # Set bais combination method
        par['calibrations']['biasframe']['process']['combine'] = 'median'
        # Set pixel flat combination method
        par['calibrations']['pixelflatframe']['process']['combine'] = 'median'

        # # Ignore PCA
        par['calibrations']['slitedges']['sync_predict'] = 'nearest'
        par['calibrations']['slitedges']['bound_detector'] = True
        # # Flats are sometimes quite ugly due to dust on the slit which leads to the erroneous detection of multiple slits. So set a higher edge_thresh and minimum_slit_gap.
        # par['calibrations']['slitedges']['edge_thresh'] = 30
        # par['calibrations']['slitedges']['minimum_slit_gap'] = 15

        # Tilt parameters
        par['calibrations']['tilts']['tracethresh'] = 25.0
        par['calibrations']['tilts']['spat_order'] = 3
        par['calibrations']['tilts']['spec_order'] = 4

        # Wavelength calibration methods
        par['calibrations']['wavelengths']['method'] = 'full_template'
        par['calibrations']['wavelengths']['lamps'] = ['NeI','HeI']
        par['calibrations']['wavelengths']['sigdetect'] = 5.
        par['calibrations']['wavelengths']['fwhm'] = 9.
        par['calibrations']['wavelengths']['cc_shift_range'] = (-100,-50)
        par['calibrations']['wavelengths']['match_toler'] = 10.
        par['calibrations']['wavelengths']['stretch_func'] = 'linear'
        par['calibrations']['wavelengths']['ech_2dfit'] = False
        par['calibrations']['wavelengths']['n_first'] = 2
        par['calibrations']['wavelengths']['n_final'] = 3

        par['reduce']['findobj']['snr_thresh'] = 300.

        # Multiple arcs with different lamps, so can't median combine nor clip, also need to remove continuum
        par['calibrations']['arcframe']['process']['clip'] = True
        par['calibrations']['arcframe']['process']['combine'] = 'median'
        par['calibrations']['arcframe']['process']['subtract_continuum'] = True

        par['calibrations']['tiltframe']['process']['clip'] = True
        par['calibrations']['tiltframe']['process']['combine'] = 'median'
        par['calibrations']['tiltframe']['process']['subtract_continuum'] = True
        par['calibrations']['standardframe']['process']['spat_flexure_correct'] = True

        # Sensitivity function and extinction file
        par['fluxcalib']['extinct_file'] = 'LJextinct.dat'
        par['fluxcalib']['extinct_correct'] = True      # 是否执行消光校正
        par['fluxcalib']['extrap_sens'] = True         # 是否外推灵敏度函数
        par['sensfunc']['UVIS']['extinct_file'] = 'LJextinct.dat'
        par['sensfunc']['use_flat'] = True
        par['sensfunc']['polyorder'] = 11
        par['sensfunc']['samp_fact'] = 0.1

        # No overscan region!
        turn_off = dict(use_overscan=False)
        par.reset_all_processimages_par(**turn_off)

        return par

    def init_meta(self):
        """
        Define how metadata are derived from the spectrograph files.
        That is, this associates the ``PypeIt``-specific metadata keywords
        with the instrument-specific header cards using :attr:`meta`.
        """
        self.meta = {}
        # Required (core)
        # self.meta['site'] = dict(ext=1, card='SITE')
        # self.meta['origname'] = dict(ext=1, card='ORIGNAME')
        self.meta['target'] = dict(ext=1, card='OBJECT')
        self.meta['ra'] = dict(ext=1, card='RA')
        self.meta['dec'] = dict(ext=1, card='DEC')
        self.meta['decker'] = dict(ext=1, card='FILTER1')
        self.meta['dispname'] = dict(ext=1, card='FILTER3')
        self.meta['mjd'] = dict(ext=1, card='MJD-OBS',rtol=1e-6)
        self.meta['dateobs'] = dict(ext=1, card='DATE-OBS')
        self.meta['exptime'] = dict(ext=1, card='EXPTIME')
        # self.meta['decker'] = dict(ext=1, card='SLIT')
        self.meta['airmass'] = dict(ext=1, card='AIRMASS')
        # Extras for config and frametyping
        self.meta['idname'] = dict(ext=1, card='OBSTYPE')
        # used for arc and continuum lamps
        self.meta['filter1'] = dict(ext=1, card='FILTER')
        self.meta['lampstat01'] = dict(ext=1, card='FILTER1')
        self.meta['lampstat02'] = dict(ext=1, card='FILTER2')
        self.meta['lampstat03'] = dict(ext=1, card='FILTER3')
        self.meta['lampstat04'] = dict(ext=1, card='FILTER4')
        self.meta['lampstat05'] = dict(ext=1, card='FILTER5')
        self.meta['lampstat06'] = dict(ext=1, card='FILTER6')
        self.meta['lampstat07'] = dict(ext=1, card='FILTER7')
        self.meta['instrument'] = dict(ext=1, card='INSTRUME')
        self.meta['binning'] = dict(card=None, compound=True)

    def compound_meta(self, headarr, meta_key):
        """
        Methods to generate metadata requiring interpretation of the header
        data, instead of simply reading the value of a header card.
        Args:
            headarr (:obj:`list`):
                List of `astropy.io.fits.Header`_ objects.
            meta_key (:obj:`str`):
                Metadata keyword to construct.
        Returns:
            object: Metadata value read from the header(s).
        """
        if meta_key == 'binning':
            binspec, binspatial = [int(item) for item in headarr[1]['CCDSUM'].split(' ')]
            return parse.binning2string(binspec, binspatial)
        elif meta_key == 'mjd':
            ttime = Time(headarr[1]['DATE-OBS'], format='isot')
            return ttime.mjd
        else:
            msgs.error("Not ready for this compound meta")

    def configuration_keys(self):
        """
        Return the metadata keys that define a unique instrument
        configuration.

        This list is used by :class:`~pypeit.metadata.PypeItMetaData` to
        identify the unique configurations among the list of frames read
        for a given reduction.

        Returns:
            :obj:`list`: List of keywords of data pulled from file headers
            and used to constuct the :class:`~pypeit.metadata.PypeItMetaData`
            object.
        """
        # return ['origname', 'filter1', 'filter2', 'filter3','filter4','filter5','filter6','filter7']
        return ['dispname','decker']

    def raw_header_cards(self):
        """
        Return additional raw header cards to be propagated in
        downstream output files for configuration identification.

        The list of raw data FITS keywords should be those used to populate
        the :meth:`~pypeit.spectrographs.spectrograph.Spectrograph.configuration_keys`
        or are used in :meth:`~pypeit.spectrographs.spectrograph.Spectrograph.config_specific_par`
        for a particular spectrograph, if different from the name of the
        PypeIt metadata keyword.

        This list is used by :meth:`~pypeit.spectrographs.spectrograph.Spectrograph.subheader_for_spec`
        to include additional FITS keywords in downstream output files.

        Returns:
            :obj:`list`: List of keywords from the raw data files that should
            be propagated in output files.
        """
        # return ['ALGRNM', 'ALAPRTNM', 'DETXBIN', 'DETYBIN']
        return ['ORIGNAME', 'FILTER1', 'FILTER2', 'FILTER3','FILTER4','FILTER5','FILTER6','FILTER7']

    def check_frame_type(self, ftype, fitstbl, exprng=None):
        """
        Check for frames of the provided type.

        Args:
            ftype (:obj:`str`):
                Type of frame to check. Must be a valid frame type; see
                frame-type :ref:`frame_type_defs`.
            fitstbl (`astropy.table.Table`_):
                The table with the metadata for one or more frames to check.
            exprng (:obj:`list`, optional):
                Range in the allowed exposure time for a frame of type
                ``ftype``. See
                :func:`pypeit.core.framematch.check_frame_exptime`.

        Returns:
            `numpy.ndarray`_: Boolean array with the flags selecting the
            exposures in ``fitstbl`` that are ``ftype`` type frames.
        """
        good_exp = framematch.check_frame_exptime(fitstbl['exptime'], exprng)
        if ftype == 'science':
            return good_exp & (fitstbl['idname'] == 'EXPOSE') & (fitstbl['lampstat01'] == 'lslit2_51') & (fitstbl['lampstat03'] == 'grism3') & (fitstbl['lampstat06'] == 'mirror_out') & (fitstbl['lampstat07'] == 'lamp_off')
        if ftype == 'standard':
            return good_exp & (fitstbl['idname'] == 'EXPOSE') & (fitstbl['lampstat01'] == 'lslit2_51') & (fitstbl['lampstat03'] == 'grism3') & (fitstbl['lampstat06'] == 'mirror_out') & (fitstbl['lampstat07'] == 'lamp_off')
        if ftype == 'bias':
            return good_exp & (fitstbl['idname'] == 'BIAS') & (fitstbl['lampstat01'] == 'diode') & (fitstbl['lampstat06'] == 'mirror_out') & (fitstbl['lampstat07'] == 'lamp_off')
        # if ftype in ['pixelflat', 'trace', 'illumflat']:
        if ftype in ['pixelflat', 'trace', 'illumflat']:
            return good_exp & (fitstbl['idname'] == 'LAMPFLAT') & (fitstbl['lampstat01'] == 'lslit2_51') & (fitstbl['lampstat03'] == 'grism3') & (fitstbl['lampstat06'] == 'mirror_in') & (fitstbl['lampstat07'] == 'lamp_halogen')
        if ftype in ['pinhole', 'dark']:
            # Don't type pinhole or dark frames
            return np.zeros(len(fitstbl), dtype=bool)
        if ftype in ['arc','tilt']:
            return good_exp & (fitstbl['idname'] == 'EXPOSE') & (fitstbl['lampstat01'] == 'lslit2_51') & (fitstbl['lampstat03'] == 'grism3') & (fitstbl['lampstat06'] == 'mirror_in') & (fitstbl['lampstat07'] == 'lamp_neon_helium')
        msgs.warn('Cannot determine if frames are of type {0}.'.format(ftype))
        return np.zeros(len(fitstbl), dtype=bool)

    def config_specific_par(self, scifile, inp_par=None):
        """
        Modify the PypeIt parameters to hard-wired values used for
        specific instrument configurations.

        Args:
            scifile (:obj:`str`):
                File to use when determining the configuration and how
                to adjust the input parameters.
            inp_par (:class:`~pypeit.par.parset.ParSet`, optional):
                Parameter set used for the full run of PypeIt.  If None,
                use :func:`default_pypeit_par`.

        Returns:
            :class:`~pypeit.par.parset.ParSet`: The PypeIt parameter set
            adjusted for configuration specific parameter values.
        """
        # Start with instrument wide
        par = super().config_specific_par(scifile, inp_par=inp_par)

        # Wavelength calibrations
        if self.get_meta_value(scifile, 'lampstat03') == 'grism3':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'ljt_yfosc_grism3_HeNe.fits'
            # par['calibrations']['wavelengths']['lamps'] = ['HeI','NeI']
        else:
            msgs.warn('ljt_yfosc.py: YOU NEED TO ADD IN THE WAVELENGTH SOLUTION FOR THIS GRISM')

        # Return
        return par