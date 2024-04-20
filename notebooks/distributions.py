import streamlit as st
import pandas as pd
import numpy as np
import warnings
import altair as alt
from scipy import stats
from scipy.stats import kstest



def best_fit_distribution(data, selected_distributions, distributions, bins=500, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x_mid = (x + np.roll(x, -1))[:-1] / 2.0  # Compute midpoints of bins

    best_distribution = None
    best_params = None
    best_sse = np.inf

    # distributions = [
    #     stats.norm, stats.expon, stats.lognorm, stats.alpha, stats.anglit, stats.erlang, stats.exponnorm,
    #     stats.uniform, stats.laplace, stats.gennorm, stats.genexpon, stats.levy, stats.loglaplace, stats.t,
    #     stats.tukeylambda, stats.pareto, stats.vonmises_line, stats.chi2, stats.gompertz
    # ]

    # Estimate distribution parameters from data

    for dist_name in selected_distributions:
        distribution = distributions[dist_name] 
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter('error', RuntimeWarning)
            try:
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                pdf = distribution.pdf(x_mid, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                

            except RuntimeWarning:
                continue
                    # Identify if this distribution is better
        if best_sse > sse > 0:
            best_distribution = distribution
            best_params = params
            best_sse = sse
    return best_distribution, best_params, best_sse

distributions = {
    'Normal': stats.norm,
    'Exponential': stats.expon,
    'Log-Normal': stats.lognorm,
    'Alpha': stats.alpha,
    'Anglit': stats.anglit,
    'Laplace': stats.laplace,
    'Uniform': stats.uniform,
    'Gennorm': stats.gennorm,
    'Levy': stats.levy,
    'LogLaplace': stats.loglaplace,
    't' : stats.t,
    'TukeyLambda': stats.tukeylambda,
    'Pareto': stats.pareto,
    'VonmisesLine': stats.vonmises_line,
    '$Chi Squared$': stats.chi2,
    'Gompertz': stats.gompertz
}

def plot_density_with_best_fit(data, best_distribution, best_params):
    # Convert data into a DataFrame for Altair
    source = pd.DataFrame({
        'Data': data
    })
    
    # Create a density plot
    density_plot = alt.Chart(source).transform_density(
        'Data',
        as_=['Data', 'Density']
    ).mark_area(opacity=0.5).encode(
        x="Data:Q",
        y='Density:Q'
    )
    
    # Generate data for the best fit line
    x_values = np.linspace(min(data), max(data), 1000)
    y_values = best_distribution.pdf(x_values, *best_params[:-2], loc=best_params[-2], scale=best_params[-1])
    best_fit_data = pd.DataFrame({'Data': x_values, 'Density': y_values})
    
    # Create the best fit line plot
    best_fit_plot = alt.Chart(best_fit_data).mark_line(color='red').encode(
        x='Data:Q',
        y='Density:Q'
    )
    
    # Combine both plots
    combined_plot = density_plot + best_fit_plot
    
    return combined_plot



st.title('Probability Distribution Fitter- Saswat K Nayak')
uploaded_file = st.file_uploader("Upload your data file", type=['dat'])
options = list(distributions.keys())
selected_distributions = st.multiselect('Select distributions to fit', options, default=options)
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file, sep=',', header=None).dropna()
        elif uploaded_file.name.endswith('.dat'):
            data = pd.read_csv(uploaded_file, sep='\s+', header=None).dropna()
        
        data = data.apply(pd.to_numeric, errors='coerce').dropna()
        


        # Iterate over columns and perform analysis
        for column in data.columns:
            st.write(f"Analysis of column: {column}")
            values = data[column].values
            if not selected_distributions:
                selected_distributions = options
            best_dist, best_params, _ = best_fit_distribution(values, selected_distributions,distributions)
            if best_dist:
                st.write(f"Best fitting distribution: {best_dist.name}")
                st.write(f"Parameters: {best_params}")
                st.altair_chart(plot_density_with_best_fit(values, best_dist, best_params), use_container_width=True)
            else:
                st.write(f"No suitable distribution found for column {column}.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.subheader("It selects the best probability distribution function for the given data as in the example below.")
    url = "https://raw.githubusercontent.com/saswatnayak1998/lyft_DE/working/notebooks/data/orders_data.dat"
    data = pd.read_csv(url, sep=',', header=None).dropna()
    data = data.apply(pd.to_numeric, errors='coerce').dropna()
    # Iterate over columns and perform analysis
    for column in data.columns:
        st.write(f"Analysis of column: {column}")
        values = data[column].values
        if not selected_distributions:
            selected_distributions = options
        best_dist, best_params, _ = best_fit_distribution(values, selected_distributions,distributions)
        if best_dist:
            st.write(f"Best fitting distribution: {best_dist.name}")
            st.write(f"Parameters: {best_params}")
            st.altair_chart(plot_density_with_best_fit(values, best_dist, best_params), use_container_width=True)
        else:
            st.write(f"No suitable distribution found for column {column}.")
        



                
    










# File uploader











import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Function definitions for fitting distributions remain unchanged

# Generate example data
def generate_example_data():
    np.random.seed(42)
    data = {
        'Gaussian': np.random.normal(loc=0, scale=1, size=1000),
        'Exponential': np.random.exponential(scale=1, size=1000),
        'Uniform': np.random.uniform(low=-2, high=2, size=1000)
    }
    return pd.DataFrame(data)

# Create interactive Altair chart
def plot_example_data(data):
    base = alt.Chart(data).transform_fold(
        ['Gaussian', 'Exponential', 'Uniform'],
        as_=['Distribution', 'Value']
    ).mark_area(
        opacity=0.5, 
        interpolate='step'
    ).encode(
        alt.X('Value:Q', bin=alt.Bin(maxbins=40), title='Value'),
        alt.Y('count()', stack=None, title='Frequency'),
        alt.Color('Distribution:N')
    ).properties(
        width=700,
        height=400
    ).interactive()

    st.altair_chart(base, use_container_width=True)


# Example data and plot button
st.subheader("Some common distributions")
example_data = generate_example_data()
plot_example_data(example_data)



