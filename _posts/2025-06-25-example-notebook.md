# Week 22 - Nonlinear Programming
---


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

sns.set_style('darkgrid')
%config InlineBackend.figure_format = 'retina'
```

## Lasso Regression

This notebook will discuss Lasso Regression, but first, we need to review multiple linear regression.

### Multiple Linear Regression

In multiple linear regression, we try to find a line of best fit with more than one predictor variable. How do we know which line is "best"? To do that, we select the line that minimizes an objective function (aka loss function). In the case of regression, this is typically the **residual sum of squares**. This minimizes the **mean squared error**.

### $$RSS = \sum_{i=1}^n (Y_i - \hat{Y}_i)^2 $$

### Minimizing the MSE
The RSS objective function has a closed form solution, which means we can solve for the beta coefficients directly without the need for iteratively searching for the best solution. We need to use linear algebra and here is the formula:

### $$ \hat{y} = X \beta$$

**Note:** $\beta$ in the formula above is a *vector* of coefficients

In different notation we could write $\hat{y}$ calculated with:

### $$ \hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n $$

$\beta$ is solved with the linear algebra formula:

### $$ \beta = (X'X)^{-1}X'y $$

Where $X'$ is the transposed matrix of original matrix $X$ and $(X'X)^-1$ is the inverted matrix of $X'X$.

---
### Back to Lasso
The closed form solution is great because it's so fast... we don't need to worry about calculating gradients and slowing moving towards to minimum. Where this can become problematic is if this soltion is _too_ good, meaning we are overfitting on our training data.

We can find the best solution on our training data, but will this generalize to the testing data? Often, the case is no. It would be better to use something like Lasso regularization to "zero out" the variables that are not important. In fact, lasso is commonly used this way. It can be a great method for feature selection.

#### How does this work?

Lasso regularization gets applied by adding and extra term to the loss function that penalizes including lots of variables.

### $$ \text{Lasso penalty}\; = \alpha \sum_{j=1}^p |\beta_j|$$

Lasso penalizes when there are many features included in the model. The \alpha term controls the strength of the penalty.

<img src="./assets/regularization.png" alt="Regularization" />

## Load the wine csv

This version has red and white wines concatenated together and tagged with a binary 1,0 indicator (1 is red wine). There are many other variables purportedly related to the rated quality of the wine.


```python
wine = pd.read_csv('./winequality_merged.csv')

# replace spaces in column names and convert all columns to lowercase:
wine.columns = [x.lower().replace(' ','_') for x in wine.columns]

wine.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>ph</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>red_wine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# separate X & y
x_cols = [x for x in wine.columns if x != 'quality']

X = wine.loc[:, x_cols].values
Y = wine['quality'].values
```


```python
# standardize
ss = StandardScaler()
X_ss = pd.DataFrame(ss.fit_transform(X), columns=x_cols)
```


```python
X_ss
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>ph</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>red_wine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.142473</td>
      <td>2.188833</td>
      <td>-2.192833</td>
      <td>-0.744778</td>
      <td>0.569958</td>
      <td>-1.100140</td>
      <td>-1.446359</td>
      <td>1.034993</td>
      <td>1.813090</td>
      <td>0.193097</td>
      <td>-0.915464</td>
      <td>1.750190</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.451036</td>
      <td>3.282235</td>
      <td>-2.192833</td>
      <td>-0.597640</td>
      <td>1.197975</td>
      <td>-0.311320</td>
      <td>-0.862469</td>
      <td>0.701486</td>
      <td>-0.115073</td>
      <td>0.999579</td>
      <td>-0.580068</td>
      <td>1.750190</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.451036</td>
      <td>2.553300</td>
      <td>-1.917553</td>
      <td>-0.660699</td>
      <td>1.026697</td>
      <td>-0.874763</td>
      <td>-1.092486</td>
      <td>0.768188</td>
      <td>0.258120</td>
      <td>0.797958</td>
      <td>-0.580068</td>
      <td>1.750190</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.073817</td>
      <td>-0.362438</td>
      <td>1.661085</td>
      <td>-0.744778</td>
      <td>0.541412</td>
      <td>-0.762074</td>
      <td>-0.986324</td>
      <td>1.101694</td>
      <td>-0.363868</td>
      <td>0.327510</td>
      <td>-0.580068</td>
      <td>1.750190</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.142473</td>
      <td>2.188833</td>
      <td>-2.192833</td>
      <td>-0.744778</td>
      <td>0.569958</td>
      <td>-1.100140</td>
      <td>-1.446359</td>
      <td>1.034993</td>
      <td>1.813090</td>
      <td>0.193097</td>
      <td>-0.915464</td>
      <td>1.750190</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6492</th>
      <td>-0.783214</td>
      <td>-0.787650</td>
      <td>-0.197054</td>
      <td>-0.807837</td>
      <td>-0.486252</td>
      <td>-0.367664</td>
      <td>-0.420128</td>
      <td>-1.186161</td>
      <td>0.320319</td>
      <td>-0.210144</td>
      <td>0.593818</td>
      <td>-0.571367</td>
    </tr>
    <tr>
      <th>6493</th>
      <td>-0.474652</td>
      <td>-0.119460</td>
      <td>0.284686</td>
      <td>0.537425</td>
      <td>-0.257883</td>
      <td>1.491697</td>
      <td>0.924588</td>
      <td>0.067824</td>
      <td>-0.426067</td>
      <td>-0.478971</td>
      <td>-0.747766</td>
      <td>-0.571367</td>
    </tr>
    <tr>
      <th>6494</th>
      <td>-0.551792</td>
      <td>-0.605417</td>
      <td>-0.885253</td>
      <td>-0.891916</td>
      <td>-0.429160</td>
      <td>-0.029599</td>
      <td>-0.083949</td>
      <td>-0.719251</td>
      <td>-1.421248</td>
      <td>-0.478971</td>
      <td>-0.915464</td>
      <td>-0.571367</td>
    </tr>
    <tr>
      <th>6495</th>
      <td>-1.323198</td>
      <td>-0.301694</td>
      <td>-0.128234</td>
      <td>-0.912936</td>
      <td>-0.971538</td>
      <td>-0.593041</td>
      <td>-0.101642</td>
      <td>-2.003251</td>
      <td>0.755710</td>
      <td>-1.016626</td>
      <td>1.935402</td>
      <td>-0.571367</td>
    </tr>
    <tr>
      <th>6496</th>
      <td>-0.937495</td>
      <td>-0.787650</td>
      <td>0.422326</td>
      <td>-0.975995</td>
      <td>-1.028631</td>
      <td>-0.480353</td>
      <td>-0.313966</td>
      <td>-1.763127</td>
      <td>0.258120</td>
      <td>-1.419867</td>
      <td>1.096912</td>
      <td>-0.571367</td>
    </tr>
  </tbody>
</table>
<p>6497 rows × 12 columns</p>
</div>



Before we build the Lasso model, let's create a plotting function so we can see the impact of Lasso regression on the beta coefficients.


```python
# the cycler package will "cycle" through colors.
from cycler import cycler

def coef_plotter(alphas, coefs, feature_names, to_alpha, regtype='lasso'):
    
    # Get the full range of alphas before subsetting to keep the plots from 
    # resetting axes each time. (We use these values to set static axes later).
    amin = np.min(alphas)
    amax = np.max(alphas)
    
    # Subset the alphas and coefficients to just the ones below the set limit
    # from the interactive widget:
    alphas = [a for a in alphas if a <= to_alpha]
    coefs = coefs[0:len(alphas)]
    
    # Get some colors from seaborn:
    colors = sns.color_palette("husl", len(coefs[0]))
    
    # Get the figure and reset the size to be wider:
    fig = plt.figure()
    fig.set_size_inches(18,5)

    # We have two axes this time on our figure. 
    # The fig.add_subplot adds axes to our figure. The number inside stands for:
    #[figure_rows|figure_cols|position_of_current_axes]
    ax1 = fig.add_subplot(121)
    
    # Give it the color cycler:
    ax1.set_prop_cycle(cycler('color', colors))
    
    # Print a vertical line showing our current alpha threshold:
    ax1.axvline(to_alpha, lw=2, ls='dashed', c='k', alpha=0.4)
    
    # Plot the lines of the alphas on x-axis and coefficients on y-axis
    ax1.plot(alphas, coefs, lw=2)
    
    # set labels for axes:
    ax1.set_xlabel('alpha', fontsize=20)
    ax1.set_ylabel('coefficients', fontsize=20)
    
    # If this is for the ridge, set this to a log scale on the x-axis:
    if regtype == 'ridge':
        ax1.set_xscale('log')
    
    # Enforce the axis limits:
    ax1.set_xlim([amin, amax])
    
    # Put a title on the axis
    ax1.set_title(regtype+' coef paths\n', fontsize=20)
    
    # Get the ymin and ymax for this axis to enforce it to be the same on the 
    # second chart:
    ymin, ymax = ax1.get_ylim()

    # Add our second axes for the barplot in position 2:
    ax2 = fig.add_subplot(122)
    
    # Position the bars according to their index from the feature names variable:
    ax2.bar(list(range(1, len(feature_names)+1)), coefs[-1], align='center', color=colors)
    ax2.set_xticks(list(range(1, len(feature_names)+1)))
    
    # Reset the ticks from numbers to acutally be the names:
    ax2.set_xticklabels(feature_names, rotation=65, fontsize=12)
    
    # enforce limits and add titles, labels
    ax2.set_ylim([ymin, ymax])
    ax2.set_title(regtype+' predictor coefs\n', fontsize=20)
    ax2.set_xlabel('coefficients', fontsize=20)
    ax2.set_ylabel('alpha', fontsize=20)
    
    plt.show()

```

Load the ipython widgets so we can make this plotting function interactive!


```python
from ipywidgets import *
from IPython.display import display
```

## Build a Lasso Regression Model


```python
from sklearn.linear_model import Lasso
```


```python
Lasso()
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Lasso()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label  sk-toggleable__label-arrow ">&nbsp;&nbsp;Lasso<a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.Lasso.html">?<span>Documentation for Lasso</span></a><span class="sk-estimator-doc-link ">i<span>Not fitted</span></span></label><div class="sk-toggleable__content "><pre>Lasso()</pre></div> </div></div></div></div>




```python
# this function will take in data and a set of alphas
# and will return a list of arrays containing the coefficients 
def lasso_coefs(X, Y, alphas):
    coefs = []
    lasso_reg = Lasso()
    for a in alphas:
        lasso_reg.set_params(alpha=a)
        lasso_reg.fit(X, Y)
        coefs.append(lasso_reg.coef_)
        
    return coefs
```


```python
l_alphas = np.arange(0.001, 0.15, 0.0025)
l_coefs = lasso_coefs(X_ss, Y, l_alphas)
```


```python
len(l_coefs)
```




    60




```python
np.arange(0.001, 0.15, 0.0025)
```




    array([0.001 , 0.0035, 0.006 , 0.0085, 0.011 , 0.0135, 0.016 , 0.0185,
           0.021 , 0.0235, 0.026 , 0.0285, 0.031 , 0.0335, 0.036 , 0.0385,
           0.041 , 0.0435, 0.046 , 0.0485, 0.051 , 0.0535, 0.056 , 0.0585,
           0.061 , 0.0635, 0.066 , 0.0685, 0.071 , 0.0735, 0.076 , 0.0785,
           0.081 , 0.0835, 0.086 , 0.0885, 0.091 , 0.0935, 0.096 , 0.0985,
           0.101 , 0.1035, 0.106 , 0.1085, 0.111 , 0.1135, 0.116 , 0.1185,
           0.121 , 0.1235, 0.126 , 0.1285, 0.131 , 0.1335, 0.136 , 0.1385,
           0.141 , 0.1435, 0.146 , 0.1485])




```python
def lasso_plot_runner(alpha=0):
    coef_plotter(l_alphas, l_coefs, X_ss.columns, alpha, regtype='lasso')

interact(lasso_plot_runner, alpha=(0.001,0.2,0.0025))
```


    interactive(children=(FloatSlider(value=0.001, description='alpha', max=0.2, min=0.001, step=0.0025), Output()…





    <function __main__.lasso_plot_runner(alpha=0)>



## Summary
As you can see, a heavier dose of Lasso will collapse the smaller coefficients to zero - effectively removing them from the model. Only the strongest betas will survive the regularization and likely leading to a model that will generalize better. Of course, a dose too strong will increase the bias too much! You need to tune your regularization strength using cross validation.


```python
import ipywidgets as widgets

def return_x(x):
    return x

widgets.interact(return_x, x=widgets.IntSlider(min=0, max=10, value=1))
```


    1





    <function __main__.return_x(x)>




```python
import ipywidgets as widgets
```


```python
help(widgets)
```

    Help on package ipywidgets:
    
    NAME
        ipywidgets - Interactive widgets for the Jupyter notebook.
    
    DESCRIPTION
        Provide simple interactive controls in the notebook.
        Each Widget corresponds to an object in Python and Javascript,
        with controls on the page.
    
        To put a Widget on the page, you can display it with IPython's display machinery::
    
            from ipywidgets import IntSlider
            from IPython.display import display
            slider = IntSlider(min=1, max=10)
            display(slider)
    
        Moving the slider will change the value. Most Widgets have a current value,
        accessible as a `value` attribute.
    
    PACKAGE CONTENTS
        _version
        comm
        embed
        tests (package)
        widgets (package)
    
    SUBMODULES
        docutils
        domwidget
        interaction
        trait_types
        util
        valuewidget
        widget
        widget_bool
        widget_box
        widget_button
        widget_color
        widget_controller
        widget_core
        widget_date
        widget_description
        widget_float
        widget_int
        widget_layout
        widget_link
        widget_media
        widget_output
        widget_selection
        widget_selectioncontainer
        widget_string
        widget_style
        widget_templates
        widget_upload
    
    FUNCTIONS
        handle_kernel = register_comm_target(kernel=None)
            Register the jupyter.widget comm target
    
        load_ipython_extension(ip)
            Set up IPython to work with widgets
    
        register_comm_target(kernel=None)
            Register the jupyter.widget comm target
    
    DATA
        __jupyter_widgets_base_version__ = '1.2.0'
        __jupyter_widgets_controls_version__ = '1.5.0'
        __protocol_version__ = '2.1.0'
        interact = <ipywidgets.widgets.interaction._InteractFactory object>
        interact_manual = <ipywidgets.widgets.interaction._InteractFactory obj...
        version_info = (7, 8, 1, 'final', 0)
        widget_serialization = {'from_json': <function _json_to_widget>, 'to_j...
    
    VERSION
        7.8.1
    
    FILE
        /opt/anaconda3/lib/python3.12/site-packages/ipywidgets/__init__.py
    
    



```python

```
