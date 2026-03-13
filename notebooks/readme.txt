Data Fusion Work: S-wave Velocity and DAS for Lithological Prediction

================================================================================

This folder contains Jupyter notebooks related to a comprehensive data fusion study 
aimed at leveraging complementary geophysical datasets to predict lithological units 
along a railway segment.

PROJECT OVERVIEW:
The work explores the integration of two key geophysical data sources:

1. S-wave Velocity Profiles: Seismic velocity measurements that provide information 
   about subsurface mechanical properties and structural layering.

2. Distributed Acoustic Sensing (DAS) Data: High-resolution strain measurements 
   collected from dark fiber (repurposed telecommunications fiber-optic cables), 
   which capture dynamic responses to seismic waves and ambient noise.

MAIN OBJECTIVE:
The primary goal is to develop machine learning models that exploit the complementary 
nature of these datasets to predict the main lithological units (rock and soil types) 
along the railway corridor. By integrating S-wave velocity information with Power 
Spectral Density (PSD) signatures computed from dark fiber DAS records, the study 
aims to enhance lithological characterization and improve subsurface understanding 
for railway engineering applications.

KEY METHODOLOGY:
- Extraction of Power Spectral Density (PSD) features from continuous DAS records 
  acquired on dark fiber infrastructure
- Integration of PSD signatures with S-wave velocity depth profiles
- Application of machine learning techniques (including XGBoost and neural operators)
  to predict depth interfaces and lithological boundaries
- Evaluation of model performance across different frequency bands and spatial scales

APPLICATIONS:
This data fusion approach has important implications for:
- Non-invasive geotechnical characterization supporting railway design and maintenance
- Efficient use of existing dark fiber infrastructure for geophysical monitoring
- Enhanced reliability of lithological predictions through multi-sensor data integration

================================================================================
