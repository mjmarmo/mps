# mps
Magnetic Positioning System

Description:
Computes location in GPS using magnetic field data. 
Geomagnetic data collected from International Geomagnetic Reference Field (IGRF) 2015 model.

Data:
File naming conventions:
- F = Total Intensity
- H = Horizontal Intensity
- Z = Vertical Intensity
- I = Inclination
- D = Declination

Improvements:
- RBF method is slow (rewrite linear algebra in C++) 
- Accurate to ~15mi radius. However, have some ideas to address this. 
- Code that processes raw data, computes magnetic vector components and creates <lat,lon,alt> <Bx,By,Bz> data file has not been uploaded. 
