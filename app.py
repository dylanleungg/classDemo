#%% Import Package
import streamlit as st
import pandas as pd
#from bokeh.plotting import figure
import plotly.express as px

#%% Dummy Chart Data
df = pd.DataFrame({
    "Company": ["A", "B", "C", "A", "B", "C"],
    "Sales": [100, 200, 90, 95, 220, 87],
    "City": ["Toronto", "Toronto", "Toronto", "Montreal", "Montreal", "Montreal"]
})



#%% Page Selector
page = st.sidebar.radio('Pick type of components',('Layout','Widgets','Charts'))

if page=="Widgets":
    st.title("Widget Demos")
    
    col1, col2 = st.columns([1,2])
    with col1:
        st.markdown("**Buttons & Checkboxes**")
        with st.echo("below"):
            st.button("OK")
            st.checkbox("Check")
    with col2:
        st.markdown("**Radio Buttons**")
        with st.echo("below"):    
            st.radio('Pick a benchmark',('S&P 500','TSX','Dow Jones'))
    
    st.markdown("**Select Box**")
    with st.echo("below"): 
        st.selectbox('Select benchmark from dropdown',('S&P 500','TSX','Dow Jones'))
    
    st.markdown("**Slider**")
    with st.echo("below"):
        emplCount = st.select_slider('Select the number of employees at your firm',
                                     options=['<10', '10-20', '20-50', '50-100', '>100'])
        st.write('The number of employees at your firm are ', emplCount)
elif page=="Layout":
    with st.echo("below"):
        st.title("This is a title")
        st.header("This is a header")
        st.subheader("This is a subheader")
        st.write("This is regular text")
        df = pd.DataFrame({'Company': ["ABC", "DEF", "GHI"],'Sales': [190, 90, 45]})
        st.write(df)
        st.table(df)
else:
    with st.echo("below"):
        st.write(df)
        #Plotly Express Bar Chart
        fig = px.bar(df, x="Company", y="Sales", color="City", barmode="group",title="Plotly Chart")
        st.plotly_chart(fig)
        #Bokeh Chart
        groupDF = df.groupby('Company').sum()
        #pB = figure(x_range=list(groupDF.index),
        #           title='Bokeh Chart',
        #           x_axis_label='Company',
        #           y_axis_label='Sales',
        #           tools="pan,wheel_zoom,box_zoom,reset,crosshair,hover")
        #pB.vbar(x=groupDF.index, top=groupDF['Sales'], width=0.9)
        #st.bokeh_chart(pB)
