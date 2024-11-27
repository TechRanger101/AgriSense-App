from django.urls import path
# from .views import SentinelDataAvailabilityView, NDVIView, NDWIView, NDMIView, CRIView, LSTView, WaterStressIndexView, CropYieldIndexView, ARVIView, CARIView, MCARIView, NDVIFView, NDWIFView, NDMIFView, CRIFView, LSTFView, WaterStressIndexForecastView, CropYieldIndexForecastView, ARVIFView, CARIFView, MCARIFView
# from .viewscm import SentinelDataAvailabilityView, NDVIView, NDWIView, NDMIView, CRIView, LSTView, WaterStressIndexView, CropYieldIndexView, ARVIView, CARIView, MCARIView, NDVIFView, NDWIFView, NDMIFView, CRIFView, LSTFView, WaterStressIndexForecastView, CropYieldIndexForecastView, ARVIFView, CARIFView, MCARIFView
from .viewscmput import SentinelDataAvailabilityView, NDVIView, NIRView, NDWIView, NDMIView, CRIView, LSTView, WaterStressIndexView, CropYieldIndexView, ARVIView, CARIView, MCARIView, NDVIFView, NDWIFView, NDMIFView, CRIFView, LSTFView, WaterStressIndexForecastView, CropYieldIndexForecastView, ARVIFView, CARIFView, MCARIFView

urlpatterns = [
    path('sentinel-data-availability/', SentinelDataAvailabilityView.as_view(), name='sentinel-data-availability'),
    path('ndvi/', NDVIView.as_view(), name='calculate_ndvi'),
    path('nir/', NIRView.as_view(), name='calculate_nir'),
    path('ndwi/', NDWIView.as_view(), name='ndwi'),
    path('ndmi/', NDMIView.as_view(), name='ndmi'),
    path('cri/', CRIView.as_view(), name='cri'),
    path('lst/', LSTView.as_view(), name='lst'), 
    path('cry/', CropYieldIndexView.as_view(), name='cry'),
    path('wst/', WaterStressIndexView.as_view(), name='wst'),
    #path('tvi/', TVIView.as_view(), name='tvi'),
    #path('evi/', EVIView.as_view(), name='evi'),
    path('dsw/', ARVIView.as_view(), name='dsw'),
    path('cpl/', CARIView.as_view(), name='cpl'),
    path('cpg/', MCARIView.as_view(), name='cpg'),    

    path('ndvif/', NDVIFView.as_view(), name='calculate_ndvif'),
    path('ndwif/', NDWIFView.as_view(), name='ndwi'),
    path('ndmif/', NDMIFView.as_view(), name='ndmi'),
    path('crif/', CRIFView.as_view(), name='cri'),
    path('lstf/', LSTFView.as_view(), name='lst'), 
    path('cryf/', CropYieldIndexForecastView.as_view(), name='cry'),
    path('wstf/', WaterStressIndexForecastView.as_view(), name='wst'),
    path('dswf/', ARVIFView.as_view(), name='dsw'),
    path('cplf/', CARIFView.as_view(), name='cpl'),
    path('cpgf/', MCARIFView.as_view(), name='cpg'),
]