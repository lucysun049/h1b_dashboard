import os
import streamlit as st
import pandas as pd
import altair as alt
from vega_datasets import data

cols_to_read = {'CASE_STATUS': str, 'DECISION_DATE': str, 'JOB_TITLE': str, 'SOC_CODE': str, 'SOC_TITLE': str, 'FULL_TIME_POSITION': str, \
                'EMPLOYER_NAME': str, 'EMPLOYER_CITY': str, 'EMPLOYER_STATE': str, 'EMPLOYER_COUNTRY': str, 'WORKSITE_CITY': str, 'WORKSITE_STATE': str,  \
                'WAGE_RATE_OF_PAY_FROM': float, 'WAGE_UNIT_OF_PAY': str, 'PREVAILING_WAGE': float, 'PW_UNIT_OF_PAY': str, 'PW_WAGE_LEVEL': str, 'H_1B_DEPENDENT': str}


def read_lca_data(years, quarters):
    dfs = {}
    for year in years:
        for quarter in quarters:
            try:
                df = pd.read_parquet(f'data/LCA_Disclosure_Data_FY{year}_Q{quarter}.parquet', columns=cols_to_read)
                dfs[f'{year}_Q{quarter}'] = df
                print(f"Successfully loaded LCA_Disclosure_Data_FY{year}_Q{quarter}.parquet with {len(df)} rows")
            except Exception as e:
                print(f"An error occurred while reading LCA_Disclosure_Data_FY{year}_Q{quarter}.parquet: {e}")
    
    return dfs


@st.cache_data
def get_lca_data(years, quarters):
    # Read data from all available years
    dfs = read_lca_data(years, quarters)

    # Concatenate all dataframes vertically
    df_combined = pd.concat([df for df in dfs.values()], axis=0, ignore_index=True)

    print(f"\nTotal combined dataframe shape: {df_combined.shape}")
    print(f"Number of datasets combined: {len(dfs)}")
    print(f"Individual dataset sizes:")
    for key, df in dfs.items():
        print(f"  {key}: {len(df)} rows")
    df_combined.reset_index(drop=True, inplace=True)
    return df_combined


@st.cache_data
def get_fips():
    df_fips = pd.read_csv('data/FIPS.csv')
    return df_fips

selected_year = st.selectbox("Select Year", options=[2024, 2023, 2022], index=0)

h1b_data = get_lca_data([selected_year], [1, 2, 3, 4])
df_fips = get_fips()
# Reset index to ensure proper alignment for boolean indexing
h1b_data = h1b_data.reset_index(drop=True)
h1b_data = h1b_data[(h1b_data['CASE_STATUS'] == 'Certified') & 
                    (h1b_data['FULL_TIME_POSITION'] == 'Y') & 
                    (h1b_data['H_1B_DEPENDENT'] == 'No')]

# Streamlit app
st.title("H1B Visa Data Dashboard")
st.markdown("This dashboard visualizes H1B visa data from [US department of Labor LCA Data](https://www.dol.gov/agencies/eta/foreign-labor/performance#:~:text=The%20data%20sets%20provide%20public,depth%20longitudinal%20research%20and%20analysis.).")

# filters: job title, soc title, employer, worksite city, worksite state
with st.sidebar:
    st.markdown("### Filter Options")
    job_title = st.text_input("Select Job Title", value="")
    if job_title:
        h1b_data = h1b_data[h1b_data['JOB_TITLE'].str.lower().str.contains(job_title.lower(), na=False)]
    soc_title = st.multiselect("Select SOC Title", options=h1b_data['SOC_TITLE'].unique(), default=[])
    employer = st.multiselect("Select Employer", options=h1b_data['EMPLOYER_NAME'].unique(), default=[])
    worksite_city = st.multiselect("Select Worksite City", options=h1b_data['WORKSITE_CITY'].unique(), default=[])
    worksite_state = st.multiselect("Select Worksite State", options=h1b_data['WORKSITE_STATE'].unique(), default=[])

# filter df with the filters when filter is not "" 
mask = pd.Series([True] * len(h1b_data), index=h1b_data.index)
if soc_title:
    mask &= h1b_data['SOC_TITLE'].isin(soc_title)
if employer:
    mask &= h1b_data['EMPLOYER_NAME'].isin(employer)
if worksite_city:
    mask &= h1b_data['WORKSITE_CITY'].isin(worksite_city)
if worksite_state:
    mask &= h1b_data['WORKSITE_STATE'].isin(worksite_state)

filtered_h1b_data = h1b_data[mask]

if len(filtered_h1b_data) == 0:
    st.markdown("No data")
    st.stop()
# Display the filtered data
st.write("Filtered H1B Data:")
filtered_h1b_data.fillna({'PW_WAGE_LEVEL': 'Unknown'}, inplace=True)
st.dataframe(filtered_h1b_data)

states = alt.topo_feature(data.us_10m.url, 'states')
# hexagon chart break down by worksite state
df_worksite_state = filtered_h1b_data.groupby('WORKSITE_STATE').agg(AVG_SALARY=('PREVAILING_WAGE', 'mean'), COUNT=('PREVAILING_WAGE', 'count')).reset_index()
df_worksite_state = pd.merge(df_worksite_state, df_fips, left_on='WORKSITE_STATE', right_on='Abbreviation', how='left', validate='many_to_one')[['WORKSITE_STATE', 'AVG_SALARY', 'COUNT', 'id', 'State']]
source = data.population_engineers_hurricanes.url
variable_list = ['AVG_SALARY', 'COUNT']

chart = alt.Chart(states).mark_geoshape().encode(
    alt.Color(alt.repeat('row'), type='quantitative'),
    tooltip=[
        alt.Tooltip('State:N', title='State'),
        alt.Tooltip('AVG_SALARY:Q', title='Avg Salary', format=',.0f'),
        alt.Tooltip('COUNT:Q', title='Count', format=',.0f')
    ]
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(df_worksite_state, 'id', variable_list + ['State'])
).project(
    type='albersUsa'
).repeat(
    row=variable_list
).resolve_scale(
    color='independent'
).interactive()

st.altair_chart(chart, use_container_width=True)



# bar chart displaying top n employers
df_employers = filtered_h1b_data.groupby('EMPLOYER_NAME').agg(AVG_SALARY=('PREVAILING_WAGE', 'mean'), COUNT=('PREVAILING_WAGE', 'count')).reset_index()
# keep only top N employers
top_n = 20
df_employers = df_employers.nlargest(top_n, 'COUNT').reset_index(drop=True)
chart = alt.Chart(df_employers).mark_bar().encode(
    y=alt.Y('EMPLOYER_NAME:N', title='Employer', sort=alt.EncodingSortField(field='COUNT', order='descending')),
    x=alt.X('COUNT:Q', title='Count'),
    color=alt.Color('AVG_SALARY:Q', title='Avg Salary', scale=alt.Scale(scheme='blues')),
    tooltip=[
        alt.Tooltip('EMPLOYER_NAME:N', title='Employer'),
        alt.Tooltip('COUNT:Q', title='Count'),
        alt.Tooltip('AVG_SALARY:Q', title='Avg Salary', format=',.0f')
    ]
).properties(
    title='H1B Visa Data by Employer'
).interactive()
st.altair_chart(chart, use_container_width=True)


# Salary Diff

df_avg_salary = filtered_h1b_data.groupby(['PW_WAGE_LEVEL', 'WORKSITE_STATE']).agg(AVG_SALARY=('PREVAILING_WAGE', 'mean'), COUNT=('PREVAILING_WAGE', 'count')).reset_index()
# heatmap displaying average salary by state and PW wage level
chart = alt.Chart(df_avg_salary).mark_rect().encode(
    x=alt.X('PW_WAGE_LEVEL:N', title='PW Wage Level'),
    y=alt.Y('WORKSITE_STATE:N', title='Worksite State'),
    color=alt.Color('AVG_SALARY:Q', title='Avg Salary', scale=alt.Scale(scheme='redblue', domainMid=1e5)),
    tooltip=[
        alt.Tooltip('PW_WAGE_LEVEL:N', title='PW Wage Level'),
        alt.Tooltip('WORKSITE_STATE:N', title='Worksite State'),
        alt.Tooltip('AVG_SALARY:Q', title='Avg Salary', format=',.0f'), 
        alt.Tooltip('COUNT:Q', title='Count', format=',.0f')
    ]
).properties(
    title='H1B Visa Average Salary by Wage Level and Worksite State'
).interactive()
st.altair_chart(chart, use_container_width=True)
