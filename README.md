Most of us probably remember learning about the protection that the ozone layer in the upper atmosphere lends us. However, ground level ozone has well documented negative effects. Those with Asthma are particularly at risk to it [[1]](https://www.epa.gov/ground-level-ozone-pollution/ground-level-ozone-basics). Denver has one of the worst levels of ground level ozone in the united states [[2]](https://www.denverpost.com/2021/06/16/denver-fortcollins-worst-cities-air-pollution/). It is well known that ozone is greatly affected by pollutants emitted by cars, power plants, refineries, etc. in the presence of sunlight [[3]](https://www.epa.gov/ground-level-ozone-pollution/ground-level-ozone-basics#:~:text=This%20happens%20when%20pollutants%20emitted,high%20levels%20during%20colder%20months.).

**Goal**: To develop an accurate linear, polynomial, and ridge polynomial regression model to predict the amount of ozone in the air.

Useful links:
[Ozone Data](https://aqs.epa.gov/aqsweb/airdata/download_files.html#Raw)

[Worst Ranking Ozone Levels](https://www.lung.org/research/sota/city-rankings/most-polluted-cities)

[Historical Weather Data](https://visualcrossing.com/)

[Weather Data Documentation](https://www.visualcrossing.com/resources/documentation/weather-data/weather-data-documentation/)

[Effect of Weekends](https://www.tandfonline.com/doi/full/10.1080/10962247.2012.749312#:~:text=In%20simple%20terms%2C%20the%20ozone,NOx\)%2C%20on%20weekends.)

**Results**: Using historical weather data paired with EPA data on ozone pollutant a polynomial regression model with R^2 of 0.76 and a MAE  of .0064 and RMSE .0085 ppm on the test set is developed. There is some amount of overfitting on the training set with this model, as the MAE and RMSE on the training set are .054 and .069 respectively. However, this overfitting is relatively mild and ridge regression does not appear to do any better on the test set.
