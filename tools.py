#!/usr/bin/env python
#-*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


"""
Tools for analysing spatial data
===================
Requires:
    * Pandas
    * GeoPandas
    * Numpy
    * shapely
    

"""


__author__ = "Juan Escamilla Mólgora"
__copyright__ = "Copyright 2017, JEM"
__license__ = "GPL"
__mantainer__ = "Juan"
__email__ ="molgor@gmail.com"



import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import scipy.spatial as sp
import scipy.special as special
import numpy as np
import matplotlib.pyplot as plt
import logging
#from external_plugins.spystats import tools as tl



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)






## These are function for handling dataframes and creating subsets.
def toGeoDataFrame(pandas_dataframe,xcoord_name,ycoord_name,srs = 'epsg:4326'):
    """
    Convert Pandas objcet to GeoDataFrame
    Inputs:
        pandas_dataframe : the pandas object to spatialise
        xcoord_name : (String) the column name of the x coordinate.
        ycoord_name : (String) the column name of the y coordinate. 
        srs : (String) the source referencing system in EPSG code.
                e.g. epsg:4326 .
    """
    data = pandas_dataframe
    data['geometry'] = data.apply(lambda z : Point(z[xcoord_name], z[ycoord_name]), axis=1)
    #data['geometry'] = data.apply(lambda z : Point(z.LON, z.LAT), axis=1)

    new_data = gpd.GeoDataFrame(data)
    new_data.crs = {'init':'epsg:4326'}
    return new_data

def _subselectDataFrameByCoordinates(dataframe,namecolumnx,namecolumny,minx,maxx,miny,maxy):
    """
    Returns a subselection by coordinates using the dataframe/
    """
    minx = float(minx)
    maxx = float(maxx)
    miny = float(miny)
    maxy = float(maxy)
    section = dataframe[lambda x:  (x[namecolumnx] > minx) & (x[namecolumnx] < maxx) & (x[namecolumny] > miny) & (x[namecolumny] < maxy) ]
    return section

def _getExtent(geodataframe):
    """
    REturn the tuple of the spatial extent. Based on geopandas geometry attribute.
    """
    minx = min(geodataframe.geometry.x)
    maxx = max(geodataframe.geometry.x)

    miny = min(geodataframe.geometry.y)
    maxy = max(geodataframe.geometry.y)
    

    return (minx,maxx,miny,maxy)
  
def _getExtentFromPoint(x,y,step_sizex,step_sizey):
    """
    Returns a tuple (4) specifying the minx,maxx miny, maxy based on a given point and a step size.
    The x,y point is located in the bottom left corner.
    """  
    minx = x
    miny = y
    maxx = x + step_sizex
    maxy = y + step_sizey
    return (minx,maxx,miny,maxy)

def _getDistanceMatrix(geopandas_dataset):
    """
    Returns the "self/auto" distance matrix of a given list of vector data. 
    By default it uses the Minkowski distance of order 2.
    Parameters :
        p : the Minkowski distance exponent (order)
    """
    data = geopandas_dataset
    coords = zip(data.centroid.x,data.centroid.y)
    dM = sp.distance_matrix(coords,coords,p=2.0)
    return dM
    
def _getDistResponseVariable(geopandas_dataset,response_variable_name):
    """ 
    Returns the "self/auto" distance matrix of a response variable Y 
    By default it uses the Minkowski distance of order 2.
    meaning:
    $$ v_{i,j} = \sum_{i=1}^{n} (y_i - y_j)^2 $$
    Parameters :
        geopandas_dataset : (geopandas) the geopandas dataframe.
        response_variable_name : (string) name of the variable for calculating the distance.
        p : the Minkowski distance exponent (order)
    """
    data = geopandas_dataset
    y = data[response_variable_name]
    yy = y.values.reshape(-1,1)
    dY = sp.distance_matrix(yy,yy,p=2.0)
    return dY
 



## This  functions are for defining an empirical variogram
def calculateEmpiricalVariogram(distances,response_variable,n_bins=50,distance_threshold=False):
    """
    Returns the empirical variogram given by the formula
    $$ v_{lag_i} = \frac{\sum_{i=1}^{N(lag_i)} (y_i - y_j)^2}{2} $$    
    Parameters:
    n_bins : (Integer) number of bins (lag distances)  
    """
    mdist = min(distances)
    if distance_threshold :
        Mdist = distance_threshold * (1.0/10.0 + 1)
    else:
        Mdist = max(distances)
        
    partitions = np.linspace(mdist,Mdist,n_bins)
    lags = partitions[:n_bins - 1]
    y = response_variable
    d = pd.DataFrame({'dist': distances,'y':y})
    
    if distance_threshold:
        d = d[ d['dist'] < distance_threshold ]
    # The actual emp. var function     
    empvar =  map(lambda (i,x) : 0.5 * (d[ ( d.dist < partitions[i+1]) & (d.dist>partitions[i])].y.mean()),enumerate(lags))
    ## Get number of elements here
    n_points =  map(lambda (i,x) : d[ ( d.dist < partitions[i+1]) & (d.dist>partitions[i])].shape[0],enumerate(lags))
   
    
    
    #self.empirical = empvar
    results = pd.DataFrame({'lags':lags,'variogram':empvar,'n_points' : n_points})
    
    return results  
 

def montecarloEnvelope(distances,response_variable,num_iterations=99,n_bins=50,distance_threshold=False):
    """
    Generate Monte Carlo envelope by shuffling the response variable and keeping the distances the same.
    After GeoR. ("Model-Based Geostatistics; Diggle and Ribeiro, 2007")
    Parameters :
        distances (List) linearised format of a distance matrix
        response_variable (list) linearised format of a response distance matrix.
    """
    simulation_variograms = []
    d = calculateEmpiricalVariogram(distances,response_variable,n_bins=n_bins,distance_threshold=distance_threshold)
    for i in range(num_iterations):
        #np.random.shuffle(response_variable)
        d = calculateEmpiricalVariogram(distances,response_variable,n_bins=n_bins,distance_threshold=distance_threshold)
        simulation_variograms.append(d.variogram)
        np.random.shuffle(response_variable)

    #simulation_variograms.append(d.lags)
    #sims = pd.DataFrame(simulation_variograms).transpose()
    sims = pd.DataFrame(simulation_variograms)
    ## Drop any possible Nan, incompatible with quantile
    sims = sims.dropna(axis=1)
    #sims.set_index('lags')
    
    low_q = sims.quantile(0.025)
    high_q = sims.quantile(0.975)
    envelope = pd.DataFrame({'envlow':low_q,'envhigh':high_q,'lags':d.lags})
    return (envelope,sims)
    #return envelope    

class Variogram(object):
    """
    A class that defines Empirical Variogram objects.
    """
    def __init__(self,geopandas_dataset,response_variable_name,model='',using_distance_threshold=False):
        """
        Constructor
            Parameters :
                geopandas_dataset : (geopandas) the geopandas dataframe.
                response_variable_name : (string) name of the variable for calculating the distance.
                p : the Minkowski distance exponent (order)
                distance_threshold : same units as coordinates
        """
        self.data = geopandas_dataset
        self.selected_response_variable = response_variable_name
        self.empirical = pd.Series()
        self.lags = []
        self.envelope = pd.DataFrame()
        self.distance_threshold = using_distance_threshold
        self.n_points = []
        self.model = model
  
    @property
    def distance_coordinates(self): 
        return _getDistanceMatrix(self.data)
    @property
    def distance_responses(self):
        return _getDistResponseVariable(self.data,self.selected_response_variable)

     
    def calculateEmpirical(self,n_bins=50):
        """
        Returns the empirical variogram given by the formula:
        $$ v_{lag_i} = \frac{\sum_{i=1}^{N(lag_i)} (y_i - y_j)^2}{2} $$
        
        Parameters:
            n_bins : (Integer) number of bins (lag distances) 
        
        This only assigns the data to the corresponding attributes.
        
        """
        
        distances = self.distance_coordinates.flatten()
        y = self.distance_responses.flatten()
        results = calculateEmpiricalVariogram(distances,y,n_bins=n_bins,distance_threshold=self.distance_threshold)
        
        self.lags = results.lags
        self.empirical = results.variogram
        self.n_points = results.n_points
        return self.empirical
        
    def calculateEnvelope(self,num_iterations=99,n_bins=50):
        """
        Calculates the Montecarlo variogram envelope.
        """
        logger.info("Calculating envelope via MonteCarlo Simulations. \n Using %s iterations"%num_iterations)
        distances = self.distance_coordinates.flatten()
        responses = self.distance_responses.flatten()
        envelopedf,sims = montecarloEnvelope(distances,responses,num_iterations=num_iterations,n_bins=n_bins,distance_threshold=self.distance_threshold)
        envelopedf = pd.concat([envelopedf,self.empirical],axis=1)
        self.envelope = envelopedf
        return envelopedf
    
    def plot(self,with_envelope=False,percentage_trunked=10,refresh=True,n_bins=50,plot_filename=False,**kwargs):
        """
        Plot the empirical semivariogram with optional confidence interval using MonteCarlo permutations at 0.025 and 0.975 quantiles.
        Returns a matplotlib object.
        Parameters : 
            with_envelope : (Boolean) if true it will calculate and plot the 0.025 and 0.975 quantiles of a montecarlo permutations of $Y$ using fixed locations.
            percentage_trunked = (float) Percentage of data removed in the plot. This is to ease the visualisation by cutting the last values
                    
        Extra parameters in the kwargs
            * num_iterations : (Integer) see CalculateEnvelope
            * n_bins : (Integer) see calculate_variogram 
        
        """
        
        

        if (self.empirical.empty or refresh == True):
            logger.info("Calculating empirical variogram")
            self.calculateEmpirical(n_bins=n_bins)
            try :
                self.envelope.variogram = self.empirical
            except:
                logger.info("Envelope attribute not found. Storing variogram in empirical attribute")
        
        nrows = self.empirical.shape[0]
        indx = int(np.ceil(float(percentage_trunked)/100 * nrows))
        
        lags = self.lags.iloc[: (nrows - indx)]
        empirical = self.empirical.iloc[:(nrows - indx)]
        
                           
        if with_envelope:
            if ( self.envelope.empty or refresh == True) :
                logger.info("No envelope object found. Calculating...")
                num_iter = kwargs.get('num_iterations')
                if isinstance(num_iter, int):
                    self.calculateEnvelope(num_iterations=num_iter,n_bins=n_bins)
                else:
                    self.calculateEnvelope()
            else:
                logger.info("Using previously stored envelope. Use refresh option to recalculate.")   
            
            envelope = self.envelope.iloc[:(nrows - indx)]
            
            plt.plot(lags,envelope.envhigh,'k--')
            plt.plot(lags,envelope.envlow,'k--')
            plt.fill_between(lags,envelope.envlow,envelope.envhigh,alpha=0.5)
            plt.legend(labels=['97.5%','emp. varig','2.5%'])
        
        
        ## ********* PLOT    
        plt.plot(lags,empirical,'o--',lw=2.0)         
        ## ****** PLOT
        plt.legend(loc='best')
        plt.xlabel("Distance in meters")
        plt.ylabel("Semivariance")
        #plt.legend(labels=['97.5%','emp. varig','2.5%'])
        #ax = 
        #points2 = plt.lines(vg.lags,vg.empirical,c='red')
        #plt.show()
        logger.debug("Check which object to return. maybe a figure")
        

        
        
        if plot_filename :
            plt.savefig(plot_filename)
        
        return None 
        


## This function is useful for calculating the empirical variogram using the chunk method
## Attention: this method does not implements a neighbouring pointa (out of the edge effect)

    def fitVariogramModel(self,model_instance,parameter_set=[]):
        """
        Fits a valid model (tools.models) to the empirical variogram object as base.
    
        Parameters:
            model_class : the class of a valid model.
            parameter_set : a tuple containing the init guesses for the parameters to be fitted from the model
    
        Returns:
            A function with the optimized parameters.
        """
        if parameter_set:
            parameter_dict = model_instance.fit(self,parameter_set)
        else:
            parameter_dict = model_instance.fit(self)
        
        
        logger.info("Added fitted model to attributes")
        self.model = model_instance
        return self.model
        
    def fitTheoreticalVariogramModel(self,model,parameter_set):
        """
        Fits a valid model (tools.model) to the empirical variogram object as base.
    
        Parameters:
            model : a Valid model (function)
            parameter_set : a tuple containing the init guesses for the parameters to be fitted from the model
    
        Returns:
            A function with the optimized parameters.
            
        notes::
            Deprecated. Use: fitVariogramModel
        """
        
        logger.warn("Deprecated. Use: fitVariogramModel")
        logger.info("Removing possible NA's")
        self.envelope = self.envelope.dropna()
        variogram = self.envelope.variogram.values
        lags = self.envelope.lags.values
        from scipy.optimize import curve_fit
        #try:
        best_params, covar_model = curve_fit(model, xdata=lags, ydata=variogram, p0=parameter_set)
        #except:
        #    logger.error("Model selected doesn´t support more than 3 parametes")
        
        teovarmodel = theoreticalVariogram(model,*best_params)
        logger.info("Adding model to the attributes space")
        setattr(self,'model',model)
        setattr(self,'model_params',best_params)
        setattr(self,'model_covar',covar_model)
    
        return (teovarmodel,{'parameters':best_params,'covar_model':covar_model})

    def calculateCovarianceMatrix(self):
        """
        Returns an evaluation of the Covariance matrix, given the model stored as attribute.
        """
        MMdist = self.distance_coordinates
        Sigma = self.model.calculateCovarianceMatrixWith(MMdist)
        return Sigma




def PartitionDataSet(geodataset,namecolumnx,namecolumny,n_chunks=10,minimmum_number_of_points=10):
    """
    Divides the given geodataset into n*n number of chunks
    Parameters : 
        geodataset (GeoDataset) with defined coordinates (geometric column)
        n_chunks : (int) desired number of chunks per dimension (resulting nxn chunks)
        minimmum_number_of_points : (int) the minimum number of points for accepting a chunk as valid.
    """
    data = geodataset
    minx,maxx,miny,maxy = _getExtent(data)
    N = n_chunks
    xp,dx = np.linspace(minx,maxx,N,retstep=True)
    yp,dy = np.linspace(miny,maxy,N,retstep=True)
    xx,yy = np.meshgrid(xp,yp)
    coordinates_list = [ (xx[i][j],yy[i][j]) for i in range(N) for j in range(N)]
    from functools import partial
    tuples = map(lambda (x,y) : partial(_getExtentFromPoint,x,y,step_sizex=dx,step_sizey=dy)(),coordinates_list)
    chunks = map(lambda (mx,Mx,my,My) : _subselectDataFrameByCoordinates(data,namecolumnx,namecolumny,mx,Mx,my,My),tuples)
    ## Here we can filter based on a threshold
    threshold = minimmum_number_of_points
    chunks_non_empty = filter(lambda df : df.shape[0] > threshold ,chunks)
    return chunks_non_empty
        

def createGrid(minx=0.0, maxx = 1.0,miny=0.0,maxy=1.0,grid_sizex=50,grid_sizey=50):
    """
    Create a Square Grid with custom coordinates an size.
    Parameters :
        minx : (Float) where to start point 0
        maxx :(FLoat) where to finish grid 
        grid_size : (Int) number of points per side.
    """
    nx = np.linspace(minx,maxx,grid_sizex)
    ny = np.linspace(miny,maxy,grid_sizey)
    xx, yy = np.meshgrid(nx,ny)
    points = pd.DataFrame({'Lon': xx.ravel(),'Lat':yy.ravel()})
    points = toGeoDataFrame(points,'Lon','Lat')
    return points



def simulateGaussianRandomField(variogram_model,grid,random_seed=False,return_matrix=True):
    """
    Returns a Random simulation of the derived Multivariate Normal Distribution with $\mu$ response variable and $Sigma$ = Resulting Covariate Matrix from model.
    Parameters : 
        variogram Model (Variogram) : an instance from variogram
        grid (Dataframe) : a geodata frame with coordinates.
        random_seed : (Integer) a random seed to reproduce results.
        
        return_matrix : (Bool) if true it will return a square matrix.
            note: For this case the grid must be SQUARE. 
        
        note: to create the variogram see: createSquareGrid
        
    """
    n_sq = len(grid)
    Y = np.zeros(n_sq)
    grid['Y'] = Y
    vg = Variogram(grid,'Y',model=variogram_model)
    logger.info("Calculating Sigma (CovMat)")
    Sigma = vg.calculateCovarianceMatrix()
    from scipy.stats import multivariate_normal as mvn
    if random_seed:
        np.random.seed(random_seed)
    sim1 = mvn.rvs(mean=Y,cov=Sigma)   
    if return_matrix:
        n = int(np.sqrt(n_sq))
        return sim1.reshape(n,n)
    else:
        s = pd.DataFrame({'sim':sim1})
        return s

def simulatedGaussianFieldAsPcolorMesh(variogram_model,minx=0.0, maxx = 1.0,miny=0.0,maxy=1.0,grid_sizex=50,grid_sizey=50,random_seed=False):
    """
    A wrapper that creates a grid and performs a simulated Gaussian process
    It returns a triplet to plot directly using matplotlib.pcolor or pcolormesh.
    
    note: It uses the functions: `createGrid` and `simulateGaussianRandomField`
    """
    grid1 = createGrid(minx, maxx, miny, maxy, grid_sizex, grid_sizey)
    sim = simulateGaussianRandomField(variogram_model,grid1,random_seed=random_seed,return_matrix=False)
    del(grid1)
    nx = np.linspace(minx,maxx,grid_sizex)
    ny = np.linspace(miny,maxy,grid_sizey)
    xx, yy = np.meshgrid(nx,ny)
    sim = sim.sim.values.reshape(grid_sizey,grid_sizex)
    return (xx,yy,sim)

### Theoretical Models
def gaussianVariogram(h,sill=0,range_a=0,nugget=0):
    """
    The Gaussian Variogram, positive SEMI definite, and it's not recommended to use.
    
    $$\gamma (h)=(s-n)\left(1-\exp \left(-{\frac  {h^{2}}{r^{2}a}}\right)\right)+n1_{{(0,\infty )}}(h)$$
    
    Parameters:
        h : (Float or Numpy Array) Distances to evaluate
        sill : Float
        range_a : Float
        nugget : Float 
        
    """
    if isinstance(h,np.ndarray):
        #Ih = np.array([1.0 if hx >= 0.0 else 0.0 for hx in h])
        Ih = np.copy(h)
        Ih[Ih >= 0.0 ] = 1.0
        Ih[Ih < 0.0] = 0.0
    else:
        Ih = 1.0 if h >= 0 else 0.0
    #Ih = 1.0 if h >= 0 else 0.0    
    g_h = ((sill - nugget)*(1 - np.exp(-(h**2 / range_a**2)))) + nugget*Ih
    return g_h

def exponentialVariogram(h,sill=0,range_a=0,nugget=0):
    """
    The exponential variogram model
    
    $$\gamma (h)=(s-n)(1-\exp(-h/(ra)))+n1_{{(0,\infty )}}(h)$$
    
        Parameters:
        h : (Float or Numpy Array) Distances to evaluate
        sill : Float
        range_a : Float
        nugget : Float 
        
    """
    
    if isinstance(h,np.ndarray):
        #Ih = np.array([1.0 if hx >= 0.0 else 0.0 for hx in h])
        Ih = np.copy(h)
        Ih[Ih >= 0.0 ] = 1.0
        Ih[Ih < 0.0] = 0.0
    else:
        Ih = 1.0 if h >= 0 else 0.0
    g_h = (sill - nugget)*(1 - np.exp(-h/range_a)) + (nugget*Ih)
    return g_h

def sphericalVariogram(h,sill=0,range_a=0,nugget=0):
    """
    The spherical variogram 
    
    $$\gamma (h)=(s-n)\left(\left({\frac  {3h}{2r}}-{\frac  {h^{3}}{2r^{3}}}\right)1_{{(0,r)}}(h)+1_{{[r,\infty )}}(h)\right)+n1_{{(0,\infty )}}(h))$$

        Parameters:
        h : (Float or Numpy Array) Distances to evaluate
        sill : Float
        range_a : Float
        nugget : Float     

    """
    
    if isinstance(h,np.ndarray):
        #Ih = np.array([1.0 if hx >= 0.0 else 0.0 for hx in h])
        Ih = np.copy(h)
        Ih[Ih >= 0.0 ] = 1.0
        Ih[Ih < 0.0] = 0.0
        I0r = np.array([1.0 if hi <= range_a else 0.0 for hi in h])
        Irinf = [1.0 if hi > range_a else 0.0 for hi in h]
    else:
        Ih = 1.0 if h >= 0 else 0.0
        I0r = [1.0 if hi <= range_a else 0.0 for hi in h]
        Irinf = [1.0 if hi > range_a else 0.0 for hi in h]
    g_h = (sill - nugget)*((3*h / float(2*range_a))*I0r + Irinf) - (h**3 / float(2*range_a)) + (nugget*Ih)
    return g_h



#def MaternVariogram(h,range_a,nugget=40,sill=100,kappa=0.5):
def maternVariogram(h,sill=1,range_a=100,nugget=40,kappa=0.5):    
    """
    The Matern Variogram of order $\kappa$.
   
    $$ \gamma(h) = nugget + (sill (1 - (\farc{1}{2^{\kappa -1}} \Gamma(\kappa) (\frac{h}{r})^{\kappa} K_\kappa \Big(\frac{h}{r}\Big)$$
    
    Let:
         a = $(2^{\kappa -1} \Gamma(\kappa))^{-1}$ 
        b = $\frac{h}{\phi}$
        K_v = Modified Bessel function of the second kind of real order v
    """
    
    #a = np.power(2, 1 - kappa) / special.gamma(kappa)
    #b = (np.sqrt(2 * kappa) / range_a) * h
    a = 1 / (np.power(2,kappa - 1 ) * special.gamma(kappa))
    
    b = (h / float(range_a))
    #Modified bessel function of second kind order kappa evaluated at b
    K_v_b = special.kv(kappa,b)
    
    rho_h = a * np.power(b,kappa) * K_v_b
    
    kh = sill * (1 - rho_h)
    
    ##legacy
    ##kh = sigma * a * np.power(b,kappa) * K_v
    ##kh = (sill - nugget) * ( 1 - (a * np.power(b,kappa) * K_v))
    #kh = nugget + (sill * ( 1 - (a * np.power(b,kappa) * K_v)))
    ## end legacy
    kh = np.nan_to_num(kh)

    return kh   

def whittleVariogram(h,sill=0,range_a=0,nugget=0,alpha=1):
    """
    The Whittle Variogram, an alternative to the Gaussian Model.
    $$\gamma (h)=(s-n)\left(1-\exp \left(-{\frac  {h^{\alpha}}{r^{\alpha}a}}\right)\right)+n1_{{(0,\infty )}}(h)$$
    
    Parameters:
        h : (Float or Numpy Array) Distances to evaluate
        sill : Float
        range_a : Float
        nugget : Float 
        
    """
    if isinstance(h,np.ndarray):
        #Ih = np.array([1.0 if hx >= 0.0 else 0.0 for hx in h])
        Ih = np.copy(h)
        Ih[Ih >= 0.0 ] = 1.0
        Ih[Ih < 0.0] = 0.0
    else:
        Ih = 1.0 if h >= 0 else 0.0
    #Ih = 1.0 if h >= 0 else 0.0    
    g_h = ((sill - nugget)*(1 - np.exp(-(h**alpha / range_a**alpha)))) + nugget*Ih
    return g_h

def theoreticalVariogram(model_function,sill,range_a,nugget,alpha=0):
    if alpha == 0:
        return lambda x : model_function(x,sill,range_a,nugget)
    else: 
        return lambda x : model_function(x,sill,range_a,nugget,alpha)

class VariogramModel(object):
    """
    This class specifies several variogram models.
    """
    def __init__(self,sill=0,range_a=0,nugget=0):
        """
        Constructor:
            optional parameters for defining the model.
        """
        self.sill = sill
        self.range_a = range_a
        self.nugget = nugget
        self.model = []
        
    def __repr__(self):
        return "< An abstract Variogram Model>"

    @property
    def f(self):
        return lambda x : self.model(x,self.sill,self.range_a,self.nugget)
    

    def corr_f(self,h):
        """
        Calculates the correlation function based on the given theoretical "intrinsic valid model"
        
        """
        ## correlation function for distances bigger than zero
        ## See: Diggle & Ribeiro (2006) section 3.5
        #corr_cont = lambda hx :  1 - (self.sill * self.f(hx) / (self.sill + self.nugget))
        ## Corrections made and suggested by Erick Chacon
        variogram_evaluated = self.f(h)
        corr_cont = (self.sill - variogram_evaluated) / (self.sill - self.nugget)
        #corr_cont = lambda hx : (self.sill - self.f(hx)) / (self.sill - self.nugget)
        #return np.array([1.0 if hx == 0 else corr_cont(hx) for hx in h])
        #corr_cont[corr_cont == 0 ] = 1.0            
        return corr_cont



    def corr_f_old(self,h):
        """
        Calculates the correlation function based on the given theoretical "intrinsic valid model"
        note:
            Deprecated : Use corr_f
        """
        ## correlation function for distances bigger than zero
        ## See: Diggle & Ribeiro (2006) section 3.5
        #corr_cont = lambda hx :  1 - (self.sill * self.f(hx) / (self.sill + self.nugget))
        ## Corrections made and suggested by Erick Chacon
        #variogram_evaluated = self.f(h)
        #corr_cont = (self.sill - variogram_evaluated) / (self.sill - self.nugget)
        corr_cont = lambda hx : (self.sill - self.f(hx)) / (self.sill - self.nugget)
        return np.array([1.0 if hx == 0 else corr_cont(hx) for hx in h])




    def calculateCovarianceMatrixWith(self,Mdist):
        """
        Returns an evaluation of the Covariance matrix, given a distance matrix.
        Parameters :  numpy Matrix
        """
        nuggetId = self.nugget * np.identity(Mdist.shape[0])
        s2 = self.sill - self.nugget
        correlMat =  np.array(self.corr_f(Mdist.flatten())).reshape(Mdist.shape)
        Sigma = (s2 * correlMat) + nuggetId
        return Sigma



    def fit(self,emp_variogram,init_params=[]):
        """
        Fits a the model to an empirical variogram object as base.
    
        Parameters:
            emp_variogram : (Variogram instance)
            parameter_set : a tuple containing the init guesses for the parameters to be fitted from the model
    
        Returns:
            A function with the optimized parameters
            
        """
        logger.info("Removing possible NA's")
        envelope = pd.DataFrame({'variogram':emp_variogram.empirical,'lags':emp_variogram.lags})
        envelope = envelope.dropna()
        variogram = envelope.variogram.values
        lags = envelope.lags.values
        from scipy.optimize import curve_fit
        if not init_params:
            init_params = [self.sill,self.range_a,self.nugget]
            if hasattr(self, 'alpha'):
                init_params.append(self.alpha)
                      
        try:
            best_params, covar_model = curve_fit(self.model, xdata=lags, ydata=variogram, p0=init_params)
        except TypeError:
            logger.error("This model  does not support more than 3 parameters")
            raise
            #return None
        try:
            s,r,n = best_params
            self.sill = s
            self.range_a = r
            self.nugget = n
        except:
            try:
                s,r,n,a = best_params
                self.sill = s
                self.range_a = r
                self.nugget = n
                self.alpha = a
            except:
                raise
        
        logger.info("Adding model to the attributes space")

        setattr(self,'model_covar',covar_model)
    
        return {'parameters':best_params,'covar_model':covar_model}

        
## Instances of Variogram        
class ExponentialVariogram(VariogramModel):
    """
    Subclass for exponential Variogram
    """
    def __init__(self,sill=0,range_a=0,nugget=0):
        super(ExponentialVariogram, self).__init__(sill,range_a,nugget)
        self.model = exponentialVariogram
        self.name = 'exponential'
    def __repr__(self):
        return u"< Exponential Variogram : sill %s, range %s, nugget %s >"%(self.sill,self.range_a,self.nugget)

class GaussianVariogram(VariogramModel):
    """
    Subclass for Gaussian Variogram
    """
    def __init__(self,sill=0,range_a=0,nugget=0):
        super(GaussianVariogram, self).__init__(sill,range_a,nugget)
        self.model = gaussianVariogram
        self.name = 'gaussian'
    def __repr__(self):
        return u"< Gaussian Variogram : sill %s, range %s, nugget %s >"%(self.sill,self.range_a,self.nugget)

class WhittleVariogram(VariogramModel):
    """
    Subclass for Gaussian Variogram
    """
    def __init__(self,sill=0,range_a=0,nugget=0,alpha=1):
        super(WhittleVariogram, self).__init__(sill,range_a,nugget)
        self.model = whittleVariogram
        self.name = 'whittle'
        self.alpha = alpha
    def __repr__(self):
        return u"< Whittle Variogram : sill %s, range %s, nugget %s, alpha%s >"%(self.sill,self.range_a,self.nugget,self.alpha)

    @property
    def f(self):
        return lambda x : self.model(x,self.sill,self.range_a,self.nugget,self.alpha)



class SphericalVariogram(VariogramModel):
    """
    Subclass for Spherical Variogram
    """
    def __init__(self,sill=0,range_a=0,nugget=0):
        super(SphericalVariogram, self).__init__(sill,range_a,nugget)
        self.model = sphericalVariogram
        self.name = 'spherical'
    def __repr__(self):
        return u"< Spherical Variogram : sill %s, range %s, nugget %s >"%(self.sill,self.range_a,self.nugget)


class MaternVariogram(VariogramModel):
    """
    Subclass for a Matern Variogram
    """   
    def __init__(self,sill=0,range_a=0,nugget=0,kappa=0.5):
        super(MaternVariogram, self).__init__(sill,range_a,nugget)
        self.kappa = kappa
        self.model = maternVariogram
        self.name = 'matern'
    def __repr__(self):
        return u"< Matern Variogram : sill %s, range %s, nugget %s, kappa %s >"%(self.sill,self.range_a,self.nugget,self.kappa)

    @property
    def f(self):
        return lambda x : self.model(x,self.sill,self.range_a,self.nugget,self.kappa)
        
if __name__ == "__main__":
    __package__ = "spystats"
    




        