# Fys-Stk-Project-1

In this respotory you will find codes used for project 1 in Fys-Stk4155. Authors are Markus Leira Asprusten, Maren Rasmussen and Arnlaug Høgås Skjæveland. 

## Description of programs
### /part_a:
#### part_a.py
    Compears the error of Lasso, Ridge, OLS regression, compearing the error with and without K-fold.
#### plot_m_variations.py
    Plots the error of OLS with increasing polynomial degree. 
#### OLS.py
    Prints β values for OLS regression and plot the regression. 
   
   
### /part_b:
  #### ridge.py
       Prints β values for Ridge regression and plot the regression. 
  #### testing_lambda.py
       Plots R² score against λ for OLS, Ridge and Lasso.
### /part_c:
   #### lasso.py
       Prints β values for Lasso regression and plot the regression. 
### /part_d:
  #### n59_e010_1arc_v3.tif
       Terrain data file
  #### read_plot_tif.py
       Plot the terrain data and terrain regression.
### /functions
  #### functions.py
        A file with many of the functions used in other programs in this project. 
        FrankeFunction(x,y), MSE(y, y_tilde), R2_Score(y, y_tilde), create_X(x, y, n = 5),
        plot_surface(x, y, z, title, show = False, cmap=cm.coolwarm, figsize = None),
        train_test_data(x_,y_,z_,i), K_fold(x,y,z,k,alpha,model,m=5, ret_std = False), 
        variance(y_tilde), bias(y, y_tilde), update_progress(job_title, progress) 
        and savefigure(name, figure = "gcf")
  #### regression.py
        A class hierarchy for the regression methodes. 
