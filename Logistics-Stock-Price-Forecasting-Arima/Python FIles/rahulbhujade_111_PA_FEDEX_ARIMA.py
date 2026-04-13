{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "# ARIMA and Seasonal ARIMA\n",
    "\n",
    "\n",
    "## Autoregressive Integrated Moving Averages\n",
    "\n",
    "The general process for ARIMA models is the following:\n",
    "* Visualize the Time Series Data\n",
    "* Make the time series data stationary\n",
    "* Plot the Correlation and AutoCorrelation Charts\n",
    "* Construct the ARIMA Model or Seasonal ARIMA based on the data\n",
    "* Use the model to make predictions\n",
    "\n",
    "Let's go through these steps!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.read_csv(r'C:\\Users\\Vishi Ved\\Downloads/FEDEX.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Stock Price</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2000-Jul</td>\n",
       "      <td>40.234375</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2000-Aug</td>\n",
       "      <td>40.531250</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2000-Sep</td>\n",
       "      <td>40.580000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2000-Oct</td>\n",
       "      <td>42.280000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2000-Nov</td>\n",
       "      <td>47.187500</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Date  Stock Price\n",
       "0  2000-Jul    40.234375\n",
       "1  2000-Aug    40.531250\n",
       "2  2000-Sep    40.580000\n",
       "3  2000-Oct    42.280000\n",
       "4  2000-Nov    47.187500"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Stock Price</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>272</th>\n",
       "      <td>2023-Mar</td>\n",
       "      <td>215.334000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>273</th>\n",
       "      <td>2023-Apr</td>\n",
       "      <td>230.252500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>274</th>\n",
       "      <td>2023-May</td>\n",
       "      <td>226.365000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>275</th>\n",
       "      <td>2023-Jun</td>\n",
       "      <td>231.840000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>276</th>\n",
       "      <td>2023-Jul</td>\n",
       "      <td>256.123333</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Date  Stock Price\n",
       "272  2023-Mar   215.334000\n",
       "273  2023-Apr   230.252500\n",
       "274  2023-May   226.365000\n",
       "275  2023-Jun   231.840000\n",
       "276  2023-Jul   256.123333"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Cost</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2000-Jul</td>\n",
       "      <td>40.234375</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2000-Aug</td>\n",
       "      <td>40.531250</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2000-Sep</td>\n",
       "      <td>40.580000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2000-Oct</td>\n",
       "      <td>42.280000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2000-Nov</td>\n",
       "      <td>47.187500</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Date       Cost\n",
       "0  2000-Jul  40.234375\n",
       "1  2000-Aug  40.531250\n",
       "2  2000-Sep  40.580000\n",
       "3  2000-Oct  42.280000\n",
       "4  2000-Nov  47.187500"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## Cleaning up the data\n",
    "df.columns=[\"Date\",\"Cost\"]\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# For dropping rows\n",
    "#df.drop(106,axis=0,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Vishi Ved\\AppData\\Local\\Temp\\ipykernel_8464\\2147723538.py:2: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.\n",
      "  df['Date']=pd.to_datetime(df['Date'])\n"
     ]
    }
   ],
   "source": [
    "# Convert Month into Datetime\n",
    "df['Date']=pd.to_datetime(df['Date'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Cost</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2000-07-01</td>\n",
       "      <td>40.234375</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2000-08-01</td>\n",
       "      <td>40.531250</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2000-09-01</td>\n",
       "      <td>40.580000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2000-10-01</td>\n",
       "      <td>42.280000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2000-11-01</td>\n",
       "      <td>47.187500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>272</th>\n",
       "      <td>2023-03-01</td>\n",
       "      <td>215.334000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>273</th>\n",
       "      <td>2023-04-01</td>\n",
       "      <td>230.252500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>274</th>\n",
       "      <td>2023-05-01</td>\n",
       "      <td>226.365000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>275</th>\n",
       "      <td>2023-06-01</td>\n",
       "      <td>231.840000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>276</th>\n",
       "      <td>2023-07-01</td>\n",
       "      <td>256.123333</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>277 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "          Date        Cost\n",
       "0   2000-07-01   40.234375\n",
       "1   2000-08-01   40.531250\n",
       "2   2000-09-01   40.580000\n",
       "3   2000-10-01   42.280000\n",
       "4   2000-11-01   47.187500\n",
       "..         ...         ...\n",
       "272 2023-03-01  215.334000\n",
       "273 2023-04-01  230.252500\n",
       "274 2023-05-01  226.365000\n",
       "275 2023-06-01  231.840000\n",
       "276 2023-07-01  256.123333\n",
       "\n",
       "[277 rows x 2 columns]"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.set_index('Date',inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Cost</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Date</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2000-07-01</th>\n",
       "      <td>40.234375</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2000-08-01</th>\n",
       "      <td>40.531250</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2000-09-01</th>\n",
       "      <td>40.580000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2000-10-01</th>\n",
       "      <td>42.280000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2000-11-01</th>\n",
       "      <td>47.187500</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 Cost\n",
       "Date                 \n",
       "2000-07-01  40.234375\n",
       "2000-08-01  40.531250\n",
       "2000-09-01  40.580000\n",
       "2000-10-01  42.280000\n",
       "2000-11-01  47.187500"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Cost</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Date</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2023-03-01</th>\n",
       "      <td>215.334000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2023-04-01</th>\n",
       "      <td>230.252500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2023-05-01</th>\n",
       "      <td>226.365000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2023-06-01</th>\n",
       "      <td>231.840000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2023-07-01</th>\n",
       "      <td>256.123333</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  Cost\n",
       "Date                  \n",
       "2023-03-01  215.334000\n",
       "2023-04-01  230.252500\n",
       "2023-05-01  226.365000\n",
       "2023-06-01  231.840000\n",
       "2023-07-01  256.123333"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Cost</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>277.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>128.218193</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>66.658981</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>37.847500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>80.440000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>105.207500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>172.110000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>312.132500</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             Cost\n",
       "count  277.000000\n",
       "mean   128.218193\n",
       "std     66.658981\n",
       "min     37.847500\n",
       "25%     80.440000\n",
       "50%    105.207500\n",
       "75%    172.110000\n",
       "max    312.132500"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 2: Visualize the Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Date'>"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Testing For Stationarity\n",
    "\n",
    "from statsmodels.tsa.stattools import adfuller"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_result=adfuller(df['Cost'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Ho: It is non stationary\n",
    "#H1: It is stationary\n",
    "\n",
    "def adfuller_test(cost):\n",
    "    result=adfuller(cost)\n",
    "    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']\n",
    "    for value, label in zip(result,labels):\n",
    "        print(label+' : '+str(value) )\n",
    "    if result[1] <= 0.05:\n",
    "        print(\"strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary\")\n",
    "    else:\n",
    "        print(\"weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary \")\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ADF Test Statistic : -0.7662978374822693\n",
      "p-value : 0.8288219637386485\n",
      "#Lags Used : 16\n",
      "Number of Observations Used : 260\n",
      "weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary \n"
     ]
    }
   ],
   "source": [
    "adfuller_test(df['Cost'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Differencing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Seasonal First Difference']=df['Cost']-df['Cost'].shift(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Cost</th>\n",
       "      <th>Seasonal First Difference</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Date</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2000-07-01</th>\n",
       "      <td>40.234375</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2000-08-01</th>\n",
       "      <td>40.531250</td>\n",
       "      <td>0.296875</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2000-09-01</th>\n",
       "      <td>40.580000</td>\n",
       "      <td>0.048750</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2000-10-01</th>\n",
       "      <td>42.280000</td>\n",
       "      <td>1.700000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2000-11-01</th>\n",
       "      <td>47.187500</td>\n",
       "      <td>4.907500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2000-12-01</th>\n",
       "      <td>42.432000</td>\n",
       "      <td>-4.755500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2001-01-01</th>\n",
       "      <td>42.232500</td>\n",
       "      <td>-0.199500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2001-02-01</th>\n",
       "      <td>43.472500</td>\n",
       "      <td>1.240000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2001-03-01</th>\n",
       "      <td>41.704000</td>\n",
       "      <td>-1.768500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2001-04-01</th>\n",
       "      <td>40.817500</td>\n",
       "      <td>-0.886500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2001-05-01</th>\n",
       "      <td>39.980000</td>\n",
       "      <td>-0.837500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2001-06-01</th>\n",
       "      <td>39.218000</td>\n",
       "      <td>-0.762000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2001-07-01</th>\n",
       "      <td>41.052500</td>\n",
       "      <td>1.834500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2001-08-01</th>\n",
       "      <td>40.912000</td>\n",
       "      <td>-0.140500</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 Cost  Seasonal First Difference\n",
       "Date                                            \n",
       "2000-07-01  40.234375                        NaN\n",
       "2000-08-01  40.531250                   0.296875\n",
       "2000-09-01  40.580000                   0.048750\n",
       "2000-10-01  42.280000                   1.700000\n",
       "2000-11-01  47.187500                   4.907500\n",
       "2000-12-01  42.432000                  -4.755500\n",
       "2001-01-01  42.232500                  -0.199500\n",
       "2001-02-01  43.472500                   1.240000\n",
       "2001-03-01  41.704000                  -1.768500\n",
       "2001-04-01  40.817500                  -0.886500\n",
       "2001-05-01  39.980000                  -0.837500\n",
       "2001-06-01  39.218000                  -0.762000\n",
       "2001-07-01  41.052500                   1.834500\n",
       "2001-08-01  40.912000                  -0.140500"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head(14)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ADF Test Statistic : -4.885281287628993\n",
      "p-value : 3.7249479984307205e-05\n",
      "#Lags Used : 15\n",
      "Number of Observations Used : 260\n",
      "strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary\n"
     ]
    }
   ],
   "source": [
    "## Again test dickey fuller test\n",
    "adfuller_test(df['Seasonal First Difference'].dropna())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Date'>"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df['Seasonal First Difference'].plot()"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "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"
    }
   },
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Auto Regressive Model\n",
    "![image.png](attachment:image.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAZAAAAEKCAYAAAA8QgPpAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAAsTAAALEwEAmpwYAAAxs0lEQVR4nO3deXxU5dXA8d/JvpGFAAk7Yd/EIAEFLQVBxQVRX6VYrGi12Fbtvtj6tljfLlpbrW/1taVu1KJIXSq2iAoSqAqWVSHsS5AtLCEhCSH7ef+Yiw7ZCMNMbmbmfD+f+eTe57l37jnMkJO7PVdUFWOMMeZsRbgdgDHGmOBkBcQYY4xPrIAYY4zxiRUQY4wxPrECYowxxidWQIwxxvjE1QIiIs+KyGER2dhEv4jI/4rIDhH5REQu8OqbISLbndeM1ovaGGMMuL8H8jwwqZn+K4F+zmsm8BSAiLQHZgEXAqOAWSKSFtBIjTHGnMbVAqKqy4FjzSwyBfireqwEUkWkM3AF8K6qHlPVIuBdmi9Exhhj/CzK7QDOoCuw12t+n9PWVHsDIjITz94L8fHxI7p37x6YSNuYuro6IiLc3sFsXZZz6Au3fKFt5Lxt27ajqtqxfntbLyDnTFVnA7MBLhgxQteuWeNyRK0jNzeXcePGuR1Gq7KcQ1+45QttI2cR2dNYe1sv5fsB712Gbk5bU+3NOlxa6dfgjDEmnLX1ArIAuNW5Gusi4LiqHgTeBi4XkTTn5PnlTluzjp2o4mRVbWAjNsaYMOHqISwReQkYB3QQkX14rqyKBlDVPwELgauAHUA5cLvTd0xE/gdY5bzVg6ra3Ml4AGrrlDfW72faqB7+TsUYY8KOqwVEVW8+Q78CdzfR9yzw7NlsLy46kuc+yOdLI7sjImezqjHGmHra+iEsv+qQFMPWQ6Ws2FnodijGGBP0wqqApMbHkJ4Yw7Mf5LsdijHGBL2wKiAi8OULe7BkyyH2FJ5wOxxjjAlqYVVAAG65qCeRIsz5sNHLmo0xxrRQ2BWQjOQ4rh7Wmb+v3ktpRbXb4RhjTNAKuwICcMclWZRW1vCcnQsxxhifhWUBGdYtlSuGZDB7+S6OnahyOxxjjAlKYVlAAH54xQDKq2p4cukOt0MxxpigFLYFpG+ndtw4ohsvrNjD/uKTbodjjDFBJ2wLCMC3J/YHgT+8u83tUIwxJuiEdQHpmhrPrRf15NW1+9h+qNTtcIwxJqiEdQEBuHt8XxJjonjk7a1uh2KMMUEl7AtIWmIMM8f25p1Nh1j7aZHb4RhjTNAI+wIC8NVLsuiQFMPDb23BMwCwMcaYM7ECAiTGRnHvpf34aPcxlm8/6nY4xhgTFKyAOG4e1YPu7eN5+K0t1NXZXogxxpyJFRBHTFQE379sAJsOlvDPDQfdDscYY9o8KyBerj2/CwMz2/HoO1uprq1zOxxjjGnTXC0gIjJJRLaKyA4Rua+R/sdEZL3z2iYixV59tV59C/wRT0SE8IPLB5BfWM4ra/b54y2NMSZkufZMdBGJBJ4ELgP2AatEZIGqbjq1jKp+12v5e4HhXm9xUlWz/R3XhEGduKBHKo8v3s71w7sSFx3p700YY0xIcHMPZBSwQ1V3qWoVMA+Y0szyNwMvBTooEeGHVwykoKSCF1bYQ6eMMaYpbhaQrsBer/l9TlsDItITyALe82qOE5HVIrJSRK7zZ2Cj+6Rzcd90/rx8FxXVtf58a2OMCRmuHcI6S9OAV1TV+7d5T1XdLyK9gfdEZIOq7qy/oojMBGYCZGRkkJub26INXpxWywc7Knlo3nuM6x597hm0srKyshbnGios59AXbvlC287ZzQKyH+juNd/NaWvMNOBu7wZV3e/83CUiuXjOjzQoIKo6G5gNkJOTo+PGjWtRcF9U5a0DH5BbUM39Xx5LdGRwXbCWm5tLS3MNFZZz6Au3fKFt5+zmb8VVQD8RyRKRGDxFosHVVCIyEEgDVni1pYlIrDPdAbgY2FR/3XMhInxnYj/yC8t56T+f+vOtjTEmJLhWQFS1BrgHeBvYDMxX1TwReVBErvVadBowT08fpGoQsFpEPgaWAg95X73lL5cO7MTo3un8YfF2Siqq/f32xhgT1Fw9B6KqC4GF9dp+Xm/+gUbW+xA4L6DB4dkLuf/qQVzzx/f5v6U7ue/KgYHepDHGBI3gOrDvgqFdU7hheFee/WA3e4+Vux2OMca0GVZAWuAHVwwgKkL46esbbLh3Y4xxWAFpgS6p8fzkqkH8e/tR5q3ae+YVjDEmDFgBaaHpo3pwUe/2PPTWFo6dqHI7HGOMcZ0VkBaKiBAenDKUE5U1/HbRFrfDMcYY11kBOQv9M9rx1UuymLdqrz0/3RgT9qyAnKVvTehHRnIsP/vHRntmiDEmrFkBOUtJsVH84toh5B0o4fHF290OxxhjXGMFxAeThnbmphHdeDJ3Bx/tKnQ7HGOMcYUVEB89cO0QeqUn8t2X13O83IY5McaEHysgPkqMjeLxadkcKavk639bQ2WNPTfEGBNerICcg2HdUvntjcNYsauQb720jqoaO6lujAkfVkDO0fXDuzFr8mDezjvEHXNWUXC8wu2QjDGmVVgB8YPbL87i4f86j1X5x7js0WW8+NGn1NXZmFnGmNBmBcRPvjSyB29/ZyxDu6bw09c38OWnV5J/9ITbYRljTMBYAfGjnumJvPi1C3nohvPI21/CpMeX81TuTjvBbowJSa4+UCoUiQjTRvVg3IBO/OyNjTy8aAvzV+/lB5cPIKtDIkmxUZysriUqUuielkBMlNVwY0xwsgISIJkpcfzl1hxytx7mwX9u4u4X1zZYJj0xhl9dP5RJQzu7EKExxpwbKyABNm5AJ8b06cCaPUUcP1lFSUUNCTGRVFbX8fyH+Xz9b2v57sT+fGtCX0TE7XCNMabFXC0gIjIJeByIBJ5W1Yfq9d8GPALsd5qeUNWnnb4ZwH877b9U1TmtErQPYqIiGN0nvUH7tdlduO/VDTy2eBuFJyqZNXkIkRFWRIwxwcG1AiIikcCTwGXAPmCViCxQ1U31Fn1ZVe+pt257YBaQAyiwxlk3qMZYj46M4Hc3DaNDuxj+vGwXh0sqefRL55MQYzuGxpi2z80zuKOAHaq6S1WrgHnAlBauewXwrqoec4rGu8CkAMUZUCLCT64cxM+vGcw7mwqY+ucVdjOiMSYouPmnblfA+wHj+4ALG1nuv0RkLLAN+K6q7m1i3a6NbUREZgIzATIyMsjNzT33yAOgN/Ct4bH86eMSLvvdEsb3iKZvagSZiRF0iBcizvL8SFlZWZvNNVAs59AXbvlC2865rR8reRN4SVUrReQuYA5w6dm8garOBmYD5OTk6Lhx4/wepL+MA64cW8pDb21mwdYjn7W3T4xhwsBOzBjTi6FdU1r0Xrm5ubTlXAPBcg594ZYvtO2c3Swg+4HuXvPd+PxkOQCq6v2wjaeB33qtO67eurl+j9AFAzLb8dztoyg6UcXOI2XsOFzGyl2F/GvDQV5Zu4/vTezPPZfaFVvGGPe5WUBWAf1EJAtPQZgGfNl7ARHprKoHndlrgc3O9NvAr0UkzZm/HPhJ4ENuPWmJMeQktienV3umjepBSUU1s97I4/fvbmPTwRJ+d9P5JMa29R1IY0woc+03kKrWiMg9eIpBJPCsquaJyIPAalVdAHxLRK4FaoBjwG3OusdE5H/wFCGAB1X1WKsn0YqS46J5dOr5DOmSzK8XbmbH4TKeumUEfTsluR2aMSZMufonrKouBBbWa/u51/RPaGLPQlWfBZ4NaIBtjIhw5xd6M7hzMve+tI4pT7zPb288n6uH2Z3sxpjWZwMxBaExfTvwr299gYGdk7n7xbXMXr7T7ZCMMWHICkiQykyJ48WvXcg1wzrz64VbePGjT90OyRgTZuwsbBCLjYrk8WnDKa2oYdaCjfTLSGJkr/Zuh2WMCRO2BxLkIiOE/502nG5pCXzjb2s4UHzS7ZCMMWHCCkgISEmI5i+3jqCiuo67X1xLdW2d2yEZY8KAFZAQ0bdTO35zw3ms+7SYR9/d5nY4xpgwYOdAQsjk87vw4c6jPJW7k5rzYk67Vd8YY/zN9kBCzC+uHcqYPuk8u7GKFTsLz7yCMcb4yApIiImJiuCpW0aQkSDc9cJqdhwudTskY0yIsgISglLio/nuiDhioiK4/flVHC2rdDskY0wIsgISojomRPD0jJEcKa3kzjmrqaiudTskY0yIsQISwrK7p/L4tOF8vK+Yb720jto6dTskY0wIsQIS4q4Ykuk8LvcQP39jI6pWRIwx/nHGy3hF5GLgAaCns7wAqqq9Axua8ZfbL87iUEklf1q2k4zkOL41oZ/bIRljQkBL7gN5BvgusAawA+lB6seTBnC4tIJH391Gp3axTBvVw+2QjDFBriUF5LiqvhXwSExAiQgP/9cwjpZV8dPXN9C9fQIX9+3gdljGmCDWknMgS0XkEREZLSIXnHoFPDLjd9GRETw1/QKyOiTy3ZfXc+xEldshGWOCWEsKyIVADvBr4PfO63eBDMoETmJsFH+8+QKKy6u564XVnKyyo5LGGN+csYCo6vhGXpf6Y+MiMklEtorIDhG5r5H+74nIJhH5RESWiEhPr75aEVnvvBb4I55wMbhLMo99KZvVe4q496W1dnmvMcYnZywgIpIiIo+KyGrn9XsRSTnXDYtIJPAkcCUwGLhZRAbXW2wdkKOqw4BXgN969Z1U1Wznde25xhNurh7WmQevHcLizYf57aItbodjjAlCLTmE9SxQCkx1XiXAc37Y9ihgh6ruUtUqYB4wxXsBVV2qquXO7Eqgmx+2axxfGd2Lr1zUkz8v38XfV+91OxxjTJCRM91YJiLrVTX7TG1nvWGRG4FJqnqnM/8V4EJVvaeJ5Z8AClT1l858DbAeqAEeUtV/NLHeTGAmQEZGxoh58+adS9hBo6ysjKSkpDMuV1OnPLqmgq3H6vj2BbEM6xi8I/y3NOdQEm45h1u+0DZyHj9+/BpVzanf3pLfFidF5BJVfR8+u7GwVZ+bKiK34DmR/0Wv5p6qul9EegPvicgGVd1Zf11VnQ3MBsjJydFx48a1Rsiuy83NpaW5jryompv/spInPi5jzu3DGd0nPbDBBcjZ5Bwqwi3ncMsX2nbOLTmE9Q3gSRHJF5E9wBPA1/2w7f1Ad6/5bk7baURkInA/cK2qfjasrKrud37uAnKB4X6IKSylJEQz984L6dk+gZkvrGbbIRsC3hhzZi25Cmu9qp4PDAPOU9XhqvqxH7a9CugnIlkiEgNMA067mkpEhgN/xlM8Dnu1p4lIrDPdAbgY2OSHmMJWWmIMz90+krjoSG5/bhWHSircDskY08Y1WUCcw0anLqX9HnAncKfX/DlR1RrgHuBtYDMwX1XzRORBETl1VdUjQBLw93qX6w4CVovIx8BSPOdArICco25pCTx320iKyqu45emPOFDcqkcqjTFBprlzIInOz3aN9PnlxgFVXQgsrNf2c6/piU2s9yFwnj9iMKcb2jWFZ2aMZOZfV3Pdkx/w6NRsLulnQ54YYxpqsoCo6p+dycWq+oF3n3Mi3YSo0X3S+fs3RvPNuWu55ZmPuPOSLH44aQCxUZFuh2aMaUNachL9jy1sMyFkYGYy/7r3C3zlop48/f5upjzxAVsL7OS6MeZzTe6BiMhoYAzQsd45j2TA/hQNA/ExkfzPdUMZP7AjP3rlEyY/8T7fmdiPr1zUk3Zx0W6HZ4xxWXPnQGLwnMCO4vTzICXAjYEMKlAKCwt5/vnnT2sbMmQII0eOpLq6mrlz5zZYJzs7m+zsbMrLy5k/f36D/pycHIYOHcrx48d5/fXXG/SPHj2aAQMGcPToUf75z3826B87diy9e/emoKCARYsWNeifMGEC3bt3Z+/evSxZsqRB/6RJk8jMzGTXrl0sX778s/bi4mLy8/O55ppr6NChA1u3bmXFihUN1r/++utJSUlh48aNrF69ukH/1KlTuXRgBr+b2J7fLN7Dbxdt5fF3NnNBSgVDkyv5/lenEhcbw6pVq8jLy2uw/m233QbAhx9+yLZt207ri46OZvr06QAsW7aM3bt3n9afkJDA1KlTAVi8eDH79u07rT85OZkbbrgBgEWLFrFlyxby8/M/609PT2fy5MkAvPnmmxQWFp62fmZmJpMmTQLgtddeo6Sk5LT+bt26MXGi5zTc/PnzKS8vP60/KyuLL37Rc2vS3Llzqa6uPq2/f//+jBkzBqDB9w78890D2tx37xR/fPcSEhJYv34969evBz7/XgNMnz6d6OjoNvHdKygoOK3fn9+9vLy8077X4P5375TmzoEsA5aJyPOquqfJdzBhITUukpu7lbDvZBQrj8WzsiieFUUJvPDLJWR3T2VQuyo61goJkTYwozHhoiVDmXQEfgQMAeJOtftrRN7WlJOTo439pROKAn336rETVSzbdpgN+0r4YMdRth4qJTYqgqvP68yNI7pxYe90IiMkYNtvTFu+YzdQwi3ncMsX2kbOIuLzUCZzgZeBa/DcgT4DOOLf8EywaZ8Yw/XDu3G9c///pgMlvPifPfxj3QFeW7efjORYrj6vC1eel8kFPdJavZgYYwKvJXsga1R1hIh84gyrjoisUtWRrRKhH2VlZemsWbPcDqNVFBcXk5qa2urbraqDbWWxbCiJZceJGGpVSIysY1C7Sga1q6RXQjWRAaolbuXspnDLOdzyhbaR8+233+7zHsipszMHReRq4ADQ3p/BmdAREwFDkysZmlxJRa2w/UQMm0tj+Ph4HKuL40mOqmVU2klGpFYQb+dLjAlqLdkDuQb4N56BD/+I5zLeX6hq0D0F0M6BuKeiupbcrUf464p8PtxZSFJsFHeN7c0dX8giIcY/Q8i3tZxbQ7jlHG75QtvI2edzIKp66vq/48B4fwdmwkNcdCSThmYyaWgmeQeO8/ji7fz+3W28sHIP37usPzeO6EZUZEvuazXGtBXN3Uj4R5oZ80pVvxWQiEzIG9Ilhdm35rA6/xi/XriZ+17bwAsr9/DIjeczuEuy2+EZY1qouT2Q8DjWY1yT06s9r35jDAs3FDBrQR7XPvE+917aj2+O70O07Y0Y0+Y1dyPhHO95EUnwej65MX4hIlw9rDNj+qTzizfzeGzxNhblFXDP+L5cNjiDmCgrJMa0VWf83ykio0VkE7DFmT9fRP4v4JGZsJKWGMMfpg1n9ldGUHKymrtfXMulv8/lzY8PcKYLPYwx7mjJn3d/AK4ACgGcpxGODWBMJoxdPiST5T8azzMzckiJj+bel9Zxz0vrKK+qcTs0Y0w9LTo+oKp76zXVBiAWYwCIjBAmDMpgwT2X8KNJA3hrw0Fu+tMKCo7bY3aNaUtaUkD2isgYQEUkWkR+gOcRtMYEVGSE8M1xfXlmxkjyj55gypPvs2bPMbfDMsY4WlJAvg7cDXQF9gPZzvw5E5FJIrJVRHaIyH2N9MeKyMtO/0ci0sur7ydO+1YRucIf8Zi2afzATrz6zTHEREVw059W8PCiLVTW2E6wMW5rtoCISCTwuKpOV9UMVe2kqreoamFz67WE895PAlcCg4GbRWRwvcXuAIpUtS/wGPCws+5gYBqeEYInAf/nvJ8JUQMzk3nr22OZmtOdp3J3cuNTKzhUYoe0jHFTs3eiq2qtiPQUkRhVrfLztkcBO1R1F4CIzAOmAJu8lpkCPOBMvwI8ISLitM9T1Upgt4jscN6v4VNrvGzdutX1IQFaS1sYgC1QOqb1ZWPt1YyZ9ToZW18jptwzOHQo59yUcMs53PIFd3OujU7gSJ+rmuxvySBEu4APRGQBcOJUo6o+eo6xdQW8T87vAy5sahlVrRGR40C6076y3rpdG9uIiMwEZoLnKWTFxcXnGHZwqK2tDd1ci1eTdHQfZSNu5cDgm0lcP4/oo9tCO+cmhFvO4ZYvuJdzTVpPTgyeisYkNLlMSwrITucVwemPtg0KqjobmA02mGKoOVRSwVefX8Xm6Bncd+0Qulfmh3zO9YXD5+wt3PKF1s+5pKKah9/awtyPPqV7+3iemj6C837X+LLNFhDnvEJ/VZ0egDj34xnh95RuTltjy+wTkSggBc/9KC1Z14S4jOQ45t81mm/PW8fP3sjjip5RfGGs2sOrjPGBqvLOpkPMeiOPw6UV3HFJFt+/vH+zo2U3exJdVWuBniIS4+9ggVVAPxHJct5/GlB/iPgFeJ6ACHAj8J56bkteAExzrtLKAvoB/wlAjKaNS4yN4s9fyeH2i3vx9p4avv63NXbToTFnQVVZvu0I1//fh9z1whpS4qN57ZsX87NrBp/xUQuunQNxzmncA7wNRALPqmqeiDwIrHaeN/IM8IJzkvwYniKDs9x8PCfca4C7nWJnwlBkhDBr8hCqjh3gpc2HmDZ7Jc/dNpL0pFi3QzOmTaqoruXjvcV8sOMo/9pwkJ1HTtA1NZ7f3HAeN47o1uLBTF09B6KqC4GF9dp+7jVdAdzUxLq/An7lz3hMcLusZzTjRw7j7hfX8qXZK3nxaxfSqV2c22EZ0yYUllUyb9Velmw+xIb9x6muVSIEcnq2566xfZgyvAuxUWd3N0RLHij1CwARSXLmy3yK3phWMHFwBnO+OoqvPr+KW5/5Dy/fNZqU+Gi3wzLGNarKgo8P8MCCPIrKqzm/eyp3XNKbnJ5pjOzVnpQE3/9/nLGAiMhQ4AWc56CLyFHgVlXN83mrxgTQRb3T+dMtI7hjziq+Nmc1f71jFHHRdp+pCT/bD5Xyq4Wbyd16hOzuqcybOYwBmf47kNSSA12zge+pak9V7Ql8H/iL3yIwJgDG9u/Io1OzWbXnGPe8uJaa2jq3QzKm1RwqqeC+Vz/hij8sZ01+Ef999SBe/cYYvxYPaNk5kERVXXpqRlVzRSTRr1EYEwCTz+9CcXkVP3sjjx+/uoHf3TQMz0AGxoSm0opqZi/fxV/+vYvaOmXGmF7ce2k/2icG4kLaFl6FJSI/w3MYC+AWPFdmGdPmfWV0LwpPVPGHxdtpnxjNT68aZEXEhKR38gr42RsbOVRSyTXDOvPDKwbQMz2wf+u3pIB8FfgF8BqgwL+dNmOCwrcn9KPoRBV/+fdu9hSW89B/DQvYX2TGtLbDpRU8sCCPhRsKGJjZjj/dMoLhPdJaZdsSTo8LzcrK0lmzZrkdRquwQedOV6ewsiieJUcSiY+sY3JGGf2Tqgj2nZFw+5zDLV9oOudahTXFcbx3JJFqFb6YXs7F6eVEBuA7ffvtt69R1Zz67S25Cutd4CZVLXbm0/CMhGvP4DBBI0JgTPuTZCVU8dqBZF7an0K3uGqGJFfSOa6GbnHVRLXs3iljXJdfHs3CQ0kcroyiV0IV12SU0SG29e+lPuMeiIisU9XhZ2oLBjaYYmhrac7VtXW8+NGn/G3lHrYf9tzWlBgTyRf6deTSQZ0YP6ATHdsFx13s4fY5h1u+cHrONbV1PPLOVv68bBfd0uL576sHccWQzICf1xMR3/ZAgDoR6aGqnzpv1BPPuRBjglJ0ZAQzxvTi1tE9OVxayYZ9x1m69TDvbTnMorwCAAZ1TiarQwLtYqOJj4kkPiaStIRosjokkdUhkR7tE4ixXRbTik5U1vDNuWtZtu0IX76wBz+7ejDxMe7e39SSAnI/8L6ILAME+ALO8zWCTWFhIc8///xpbUOGDGHkyJFUV1czd+7cButkZ2eTnZ1NeXk58+fPb9Cfk5PD0KFDOX78OK+//nqD/tGjRzNgwACOHj3KP//5zwb9Y8eOpXfv3hQUFLBo0aIG/RMmTKB79+7s3buXJUuWNOifNGkSmZmZ7Nq1i+XLl3/WXlxcTH5+Ptdccw0dOnRg69atrFjR8Hlb119/PSkpKWzcuJHG9s6mTp1KQkIC69evZ/369Q36p0+fTnR0NKtWrSIvr+G9pbfddhsAH374Idu2bTutLzo6munTPQM9L1u2jN27d5/Wn5CQwNSpUwFYvHgx+/btO60/OTmZG264AYBFixaxZcsW8vPzP+tPT09n8uTJALz55psUFp7+IM3MzEwmTZpExuA4SrZ8QN/MEgrSItleFsuekirWHo9Fo2Ior6qlvLKaWv38rzxBSYqJoGNyAh2SYpGSg3SKqSIztoaM2BqiIqB///6MGTMGoMH3Dvzz3QPa3HfvlEB89059r6FtffcKCgpO62/pdw/gtddeo6Sk5LT+bt26MXHiRADy8vLYviufuftS2FMezeTMMq5of+yz4jF37lyqq6tPWz/Q371TWjKUySIRuQC4yGn6jqoePdN6xgQbEegcV0vnuHLg9P/E8+fPp7D0JIVVkZ+9IhNSSExLpuB4BRuKY6is8xz2ihSlc2wNgyvLWV+1g2FdU6mpw86xGJ/UKbx6IJn88mhu6FzKsJRKt0P6TIuuwhKRa4Gxzmyuqjb8cyYI2DmQ0OZmznV1yt6icvIOlPDx3mLWflrEloOllFZ6hpZPiY/mlot6cM/4fn497BBun3O45auqfO2pd1j8aQ2zJg/m9ouzXInD53MgIvIQMBI4tZ/zbREZo6o/9XOMxgStiAihZ3oiPdMTueq8zp+1l1RUs2r3MV5bu58nl+7kX58cZPatOfTPCLqHexoXzF6+i8Wf1nDnJVmuFY/mtGSn+irgMlV9VlWfBSYB1wQ2LGNCQ3JcNBMGZfDk9At46WsXcaKqluuf/IB38grOvLIJW6rKI29v4TdvbWFkZiQ/vWqQ2yE1qqVHZVO9plMCEIcxIW90n3TevOcS+nZKYuYLa3hy6Q7C6UZe0zKVNbV8f/7HPLl0JzeP6s7Xh8US0UYf09ySAvIbYJ2IPC8ic4A1wK8DG5YxoSkzJY6X7xrNddldeOTtrXx//sdU1tjDNI1H3oHjTHniA15bt58fXN6fX19/HpFttHhAy67CeklEcvGcBwH4sara/rcxPoqLjuSxL2XTp2MSv393GzuOlPHIjef7fahtEzxUlWfe383Di7aQmhDDs7flcOnADLfDOqMz7oGIyBJVPaiqC5xXgYg0vCj8LIhIexF5V0S2Oz8bjPwlItkiskJE8kTkExH5klff8yKyW0TWO6/sc4nHmNYmItw7oR9/umUE+4tOcs0f/83ji7dTUW17I+GmpraOBxbk8ct/bWbCwAze+c7YoCge0MweiIjEAQlAB+cX/Kn9qGSg6zlu9z5giao+JCL3OfM/rrdMOZ4nH24XkS7AGhF5+9SYXMAPVfWVc4zDGFdNGprJyF5p/OLNTTy2eBvzV+/lR5MGMHlYlzZ73DuU1dTWsWZPEcdPVqN49gwiREiOj6ZOlcSYKNISYkhJiCYpNuqcDy+VVFRz74vrWLbtCF/7QhY/uXJQUH3uzR3Cugv4DtAFWOvVXgI8cY7bnQKMc6bnALnUKyCqus1r+oCIHAY6AsXnuG1j2pT0pFj+9+bhTBvZnV8t3My3563n2fd38+NJAxndJ92eX9IKik5U8cLKPbywcg9HSlt+o15CTCRJsVF0To2nU7tYoiOFqIgIoiKF6IgI+nRK5NKBnejTManB57hh33G+O389+UdP8JsbzuPmUT38nVbAtWQwxXtV9Y9+3ahIsaqmOtMCFJ2ab2L5UXgKzRBVrROR54HRQCWwBLhPVRv91EVkJs7QKxkZGSPmzZvnx0zarrKyMpKSktwOo1WFQs51qqw4UMOr26s5VqFkJUdw04AYBqc3fvNhKOR8Nvyd7+HyOt7Or+bf+2qoqoNhHSL5QrcoOsYLIp7DLnUK5TWe6cpapbRKKa+BkzXqvODoyTpKq6BWldo6qKmDGoXjlZ7fr+lxQv+0CDolRKDA9qJaNh+ro10MfPP8OAY18fkGImdfjB8/vtEbCVtSQG5trF1V/3qG9RYDmY103Q/M8S4YIlKkqo0+AUVEOuPZQ5mhqiu92gqAGDzPbN+pqg82mwh2J3qoC6WcK6preX3dfp54bwf7i08yvEcqd1ySxaQhmURFfn7q0s2cVZWCkgqOllaRFBdFelIM7WKjArrH5K98P95bzOzlu3hr40EiI4TrsrvytbG9/X6D54Hikyzdeph/bzvK+r3FFJRUANC3UxLXnt+F2y/uRbu46Gbfoy18r89lNN6RXtNxwAQ8h7SaLSCqOrGZYA6JSGdVPegUg8NNLJcM/Au4/1TxcN77oDNZKSLPAT9oQR7GBI246EhuHtWD64d35eVVe3nug93c8+I6uqbGM2NMT740sgcp8c3/4gmETQdK+NeGA6zZU0TegRJKK2pO64+JjKBzahznd0vl0oGduKh3OhnJsW3iMNyR0kqWbTvC31fv5aPdx2gXF8XMsX24/eJeZCTHBWSbXVLjmX5hT6Zf2BPwPEogUiSoznM0pyWX8d7rPS8iqcC5HgdaAMwAHnJ+vlF/ARGJAV4H/lr/ZLlX8RHgOmDjOcZjTJsUFx3JjDG9uOWinry35TDPvL+LXy/cwuOLt3NTTneyqEVVA/oLWlX5cGchTy7dwYc7C4mMEIZ2Seba87swsHMyndrFcqKyhsKyKgpPVPHpsRN8uLOQBR8fACAqQmgX5zn5nJkSR+eUeLqken52To2jizN/pr/EfbH3WDl/X72X97YeZuN+z4i3XVM9z9GYNqoHSbEt+Rvaf6IjQ2tETV/+9U4Avc9xuw8B80XkDmAPMBVARHKAr6vqnU7bWCBdRG5z1rtNVdcDc0WkI57DkuuBr59jPMa0aZERwmWDM7hscAYb9x/n2Q92M/ejPVTXKnO2LWPCoE4M75HG0C4ptIuLIj4mktioiAaFRVU5VFLJloIS9hWdJEKEqAghJSGajOQ4OrWLJSU+msOllazfW8Sq/CKWbzvCvqKTdGoXy0+uHMhNOd3P+Ez5ujplw/7jrPu0iMOllZRW1HDsRBUHj5/kw51HOVRSQV29o+dpCdHOeGIJ9ExPpFd6wmfT6YkxjRZJVWXX0RNsKyjlWHkVRSeqKCqvpuhEFdsPl7HxwHEEGNEzjR9eMYAv9u/I4M7JIbMH4LaWnAN5k88fIBUJDALmq+p9AY7N7+wcSGgLt5yLTlTx+GvL2FbRjtX5RVTV1jVYJsr5RXnqP7CqNvjF3Zx2sVFc2Ls9VwzJZPL5XYiL9s9IwjW1dRwureTg8ZMcKK5gf/FJPj1Wzp7CE+wpLOdA8cnT4oyMEFLio4nRajp3SCElPpqTVbVsPVRKcfnpz8JIiIkkLSGGbmnxXNK3AzfldCczJTCHqFpDW/hen8s5kN95TdfgKSJfamJZY0wrSUuMYVz3aB4YdxGVNbVsPljK1oISyqtqqaiu42R1LTW1dZz6w12cW7k6totlQGY7eqUnIuI5Ll90oprDpRUcKqnk+Mlq0hNjGNo1hQGZ7QIylEZUZARdUuPpkhrPiJ4N+ytratlXdJJPC8vJLzxBYVkVxSer2Ja/n5iYKI6dqCI2KoLLB2cwomcaQ7qk0LFdLKkJ0cRGufuUvnDSknMgy0RkOPBl4CZgN/BqoAMzxrRcbFQk2d1Tye6e6tP63dKgLY2TGhsVSZ+OSfTpePrlq7m5hYwbd6FLUZn6mrsTvT9ws/M6CryM55DX+FaKzRhjTBvW3B7IFuDfwDWqugNARL7bKlEZY4xp85q7puwG4CCwVET+IiIT+Hw8LGOMMWGuyQKiqv9Q1WnAQGApnnGxOonIUyJyeSvFZ4wxpo06410tqnpCVV9U1clAN2AdDUfONcYYE2bO6rZIVS1S1dmqOiFQARljjAkOoXVfvTHGmFZjBcQYY4xPrIAYY4zxiRUQY4wxPrECYowxxidWQIwxxvjECogxxhifWAExxhjjEysgxhhjfGIFxBhjjE9cKSAi0l5E3hWR7c7PtCaWqxWR9c5rgVd7loh8JCI7RORlEWn+Ac3GGGP8zq09kPuAJaraD1jizDfmpKpmO69rvdofBh5T1b5AEXBHYMM1xhhTn1sFZAowx5meA1zX0hVFRIBLgVd8Wd8YY4x/iKq2/kZFilU11ZkWoOjUfL3laoD1QA3wkKr+Q0Q6ACudvQ9EpDvwlqoObWJbM4GZABkZGSPmzZvn/4TaoLKyMpKSks68YAixnENfuOULbSPn8ePHr1HVnPrtzT3S9pyIyGIgs5Gu+71nVFVFpKkq1lNV94tIb+A9EdkAHD+bOFR1NjAbICcnR8eNG3c2qwet3NxcwiXXUyzn0Bdu+ULbzjlgBURVJzbVJyKHRKSzqh4Ukc7A4SbeY7/zc5eI5ALDgVeBVBGJUtUaPA+52u/3BIwxxjTLrXMgC4AZzvQM4I36C4hImojEOtMdgIuBTeo55rYUuLG59Y0xxgSWWwXkIeAyEdkOTHTmEZEcEXnaWWYQsFpEPsZTMB5S1U1O34+B74nIDiAdeKZVozfGGBO4Q1jNUdVCoMFjcVV1NXCnM/0hcF4T6+8CRgUyRmOMMc2zO9GNMcb4xAqIMcYYn1gBMcYY4xMrIMYYY3xiBcQYY4xPrIAYY4zxiRUQY4wxPrECYowxxidWQIwxxvjECogxxhifWAExxhjjEysgxhhjfGIFxBhjjE+sgBhjjPGJFRBjjDE+sQJijDHGJ1ZAjDHG+MQKiDHGGJ+4UkBEpL2IvCsi252faY0sM15E1nu9KkTkOqfveRHZ7dWX3do5GGNMuHNrD+Q+YImq9gOWOPOnUdWlqpqtqtnApUA58I7XIj881a+q61shZmOMMV7cKiBTgDnO9BzgujMsfyPwlqqWBzIoY4wxLedWAclQ1YPOdAGQcYblpwEv1Wv7lYh8IiKPiUis3yM0xhjTLFHVwLyxyGIgs5Gu+4E5qprqtWyRqjY4D+L0dQY+AbqoarVXWwEQA8wGdqrqg02sPxOYCZCRkTFi3rx5PucUTMrKykhKSnI7jFZlOYe+cMsX2kbO48ePX6OqOfXbowK1QVWd2FSfiBwSkc6qetApBoebeaupwOuniofz3qf2XipF5DngB83EMRtPkSEnJ0fHjRt3FlkEr9zcXMIl11Ms59AXbvlC287ZrUNYC4AZzvQM4I1mlr2ZeoevnKKDiAie8ycb/R+iMcaY5rhVQB4CLhOR7cBEZx4RyRGRp08tJCK9gO7AsnrrzxWRDcAGoAPwy9YI2hhjzOcCdgirOapaCExopH01cKfXfD7QtZHlLg1kfMYYY87M7kQ3xhjjEysgxhhjfGIFxBhjjE+sgBhjjPGJFRBjjDE+sQJijDHGJ1ZAjDHG+MQKiDHGGJ9YATHGGOMTKyDGGGN8YgXEGGOMT6yAGGOM8YkVEGOMMT6xAmKMMcYnVkCMMcb4xAqIMcYYn1gBMcYY4xMrIMYYY3xiBcQYY4xPXCkgInKTiOSJSJ2I5DSz3CQR2SoiO0TkPq/2LBH5yGl/WURiWidyY4wxp7i1B7IRuAFY3tQCIhIJPAlcCQwGbhaRwU73w8BjqtoXKALuCGy4xhhj6nOlgKjqZlXdeobFRgE7VHWXqlYB84ApIiLApcArznJzgOsCFqwxxphGRbkdQDO6Anu95vcBFwLpQLGq1ni1d23qTURkJjDTmS0TkTMVrlDRATjqdhCtzHIOfeGWL7SNnHs21hiwAiIii4HMRrruV9U3ArXd+lR1NjC7tbbXVojIalVt8vxSKLKcQ1+45QttO+eAFRBVnXiOb7Ef6O41381pKwRSRSTK2Qs51W6MMaYVteXLeFcB/ZwrrmKAacACVVVgKXCjs9wMoNX2aIwxxni4dRnv9SKyDxgN/EtE3nbau4jIQgBn7+Ie4G1gMzBfVfOct/gx8D0R2YHnnMgzrZ1DEAi7w3ZYzuEg3PKFNpyzeP6gN8YYY85OWz6EZYwxpg2zAmKMMcYnVkBChIjki8gGEVkvIqudtvYi8q6IbHd+prkd57kQkWdF5LCIbPRqazRH8fhfZ7ibT0TkAvci900T+T4gIvudz3m9iFzl1fcTJ9+tInKFO1GfGxHpLiJLRWSTM9zRt532kPycm8k3OD5nVbVXCLyAfKBDvbbfAvc50/cBD7sd5znmOBa4ANh4phyBq4C3AAEuAj5yO34/5fsA8INGlh0MfAzEAlnATiDS7Rx8yLkzcIEz3Q7Y5uQWkp9zM/kGxedseyChbQqeoV4gBIZ8UdXlwLF6zU3lOAX4q3qsxHPvUOdWCdRPmsi3KVOAeapaqaq7gR14hgMKKqp6UFXXOtOleK7A7EqIfs7N5NuUNvU5WwEJHQq8IyJrnOFbADJU9aAzXQBkuBNaQDWVY2ND4TT3HzOY3OMcrnnW67BkyOUrIr2A4cBHhMHnXC9fCILP2QpI6LhEVS/AM3rx3SIy1rtTPfu/IX3NdjjkCDwF9AGygYPA712NJkBEJAl4FfiOqpZ494Xi59xIvkHxOVsBCRGqut/5eRh4Hc9u7aFTu/POz8PuRRgwTeXY1FA4QU1VD6lqrarWAX/h88MXIZOviETj+WU6V1Vfc5pD9nNuLN9g+ZytgIQAEUkUkXanpoHL8TxzZQGeoV4gdId8aSrHBcCtzlU6FwHHvQ6BBK16x/evx/M5gyffaSISKyJZQD/gP60d37lyHtfwDLBZVR/16grJz7mpfIPmc3b7KgR7nfsL6I3nyoyPgTw8Ix6DZ5iXJcB2YDHQ3u1YzzHPl/DszlfjOfZ7R1M54rkq50k8V6lsAHLcjt9P+b7g5PMJnl8mnb2Wv9/Jdytwpdvx+5jzJXgOT30CrHdeV4Xq59xMvkHxOdtQJsYYY3xih7CMMcb4xAqIMcYYn1gBMcYY4xMrIMYYY3xiBcQYY4xPrIAY0wpEpMztGIzxNysgxhhjfGIFxBiXiMhkEflIRNaJyGIRyXDaOzrPvMgTkadFZI+IdHA7XmPqswJijHveBy5S1eHAPOBHTvss4D1VHQK8AvRwKT5jmhXldgDGhLFuwMvOuEcxwG6n/RI84x+hqotEpMil+Ixplu2BGOOePwJPqOp5wF1AnMvxGHNWrIAY454UPh+Ke4ZX+wfAVAARuRwI6mfZm9Blgyka0wpEpA444NX0KJ4RVR8DioD3gJGqOk5EOuEZiTcDWAFcA/RS1crWjdqY5lkBMaaNEZFYoFZVa0RkNPCUqma7HJYxDdhJdGPanh7AfBGJAKqAr7kcjzGNsj0QY4wxPrGT6MYYY3xiBcQYY4xPrIAYY4zxiRUQY4wxPrECYowxxif/DzsW03DUNHpJAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from pandas.plotting import autocorrelation_plot\n",
    "autocorrelation_plot(df['Cost'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Final Thoughts on Autocorrelation and Partial Autocorrelation\n",
    "\n",
    "* Identification of an AR model is often best done with the PACF.\n",
    "    * For an AR model, the theoretical PACF “shuts off” past the order of the model.  The phrase “shuts off” means that in theory the partial autocorrelations are equal to 0 beyond that point.  Put another way, the number of non-zero partial autocorrelations gives the order of the AR model.  By the “order of the model” we mean the most extreme lag of x that is used as a predictor.\n",
    "    \n",
    "    \n",
    "* Identification of an MA model is often best done with the ACF rather than the PACF.\n",
    "    * For an MA model, the theoretical PACF does not shut off, but instead tapers toward 0 in some manner.  A clearer pattern for an MA model is in the ACF.  The ACF will have non-zero autocorrelations only at lags involved in the model.\n",
    "    \n",
    "   "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# p,d,q\n",
    "    p = AR model lags\n",
    "    d = differencing (How many Seasonal differencing I did?)\n",
    "    q = MA lags"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "from statsmodels.graphics.tsaplots import plot_acf,plot_pacf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Vishi Ved\\AppData\\Roaming\\Python\\Python310\\site-packages\\statsmodels\\graphics\\tsaplots.py:348: FutureWarning: The default method 'yw' can produce PACF values outside of the [-1,1] interval. After 0.13, the default will change tounadjusted Yule-Walker ('ywm'). You can use this method now by setting method='ywm'.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 864x576 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize=(12,8))\n",
    "ax1 = fig.add_subplot(211)\n",
    "fig = plot_acf(df['Seasonal First Difference'].iloc[7:],lags=101,ax=ax1)\n",
    "ax2 = fig.add_subplot(212)\n",
    "fig = plot_pacf(df['Seasonal First Difference'].iloc[7:],lags=101,ax=ax2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Vishi Ved\\AppData\\Roaming\\Python\\Python310\\site-packages\\statsmodels\\tsa\\base\\tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.\n",
      "  self._init_dates(dates, freq)\n",
      "C:\\Users\\Vishi Ved\\AppData\\Roaming\\Python\\Python310\\site-packages\\statsmodels\\tsa\\base\\tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.\n",
      "  self._init_dates(dates, freq)\n",
      "C:\\Users\\Vishi Ved\\AppData\\Roaming\\Python\\Python310\\site-packages\\statsmodels\\tsa\\base\\tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.\n",
      "  self._init_dates(dates, freq)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>SARIMAX Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>         <td>Cost</td>       <th>  No. Observations:  </th>    <td>277</td>   \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>            <td>ARIMA(2, 1, 3)</td>  <th>  Log Likelihood     </th> <td>-1024.351</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>            <td>Sun, 20 Aug 2023</td> <th>  AIC                </th> <td>2060.702</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                <td>22:25:07</td>     <th>  BIC                </th> <td>2082.424</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Sample:</th>             <td>07-01-2000</td>    <th>  HQIC               </th> <td>2069.418</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th></th>                   <td>- 07-01-2023</td>   <th>                     </th>     <td> </td>    \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>     <td> </td>    \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>ar.L1</th>  <td>   -1.5494</td> <td>    0.027</td> <td>  -56.454</td> <td> 0.000</td> <td>   -1.603</td> <td>   -1.496</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>ar.L2</th>  <td>   -0.9598</td> <td>    0.034</td> <td>  -28.104</td> <td> 0.000</td> <td>   -1.027</td> <td>   -0.893</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>ma.L1</th>  <td>    1.8088</td> <td>    0.052</td> <td>   34.776</td> <td> 0.000</td> <td>    1.707</td> <td>    1.911</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>ma.L2</th>  <td>    1.3834</td> <td>    0.083</td> <td>   16.631</td> <td> 0.000</td> <td>    1.220</td> <td>    1.546</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>ma.L3</th>  <td>    0.2997</td> <td>    0.052</td> <td>    5.818</td> <td> 0.000</td> <td>    0.199</td> <td>    0.401</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>sigma2</th> <td>   97.7538</td> <td>    4.127</td> <td>   23.685</td> <td> 0.000</td> <td>   89.664</td> <td>  105.843</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "  <th>Ljung-Box (L1) (Q):</th>     <td>0.08</td>  <th>  Jarque-Bera (JB):  </th> <td>562.26</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(Q):</th>                <td>0.77</td>  <th>  Prob(JB):          </th>  <td>0.00</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Heteroskedasticity (H):</th> <td>16.01</td> <th>  Skew:              </th>  <td>-0.81</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(H) (two-sided):</th>    <td>0.00</td>  <th>  Kurtosis:          </th>  <td>9.80</td> \n",
       "</tr>\n",
       "</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step)."
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                               SARIMAX Results                                \n",
       "==============================================================================\n",
       "Dep. Variable:                   Cost   No. Observations:                  277\n",
       "Model:                 ARIMA(2, 1, 3)   Log Likelihood               -1024.351\n",
       "Date:                Sun, 20 Aug 2023   AIC                           2060.702\n",
       "Time:                        22:25:07   BIC                           2082.424\n",
       "Sample:                    07-01-2000   HQIC                          2069.418\n",
       "                         - 07-01-2023                                         \n",
       "Covariance Type:                  opg                                         \n",
       "==============================================================================\n",
       "                 coef    std err          z      P>|z|      [0.025      0.975]\n",
       "------------------------------------------------------------------------------\n",
       "ar.L1         -1.5494      0.027    -56.454      0.000      -1.603      -1.496\n",
       "ar.L2         -0.9598      0.034    -28.104      0.000      -1.027      -0.893\n",
       "ma.L1          1.8088      0.052     34.776      0.000       1.707       1.911\n",
       "ma.L2          1.3834      0.083     16.631      0.000       1.220       1.546\n",
       "ma.L3          0.2997      0.052      5.818      0.000       0.199       0.401\n",
       "sigma2        97.7538      4.127     23.685      0.000      89.664     105.843\n",
       "===================================================================================\n",
       "Ljung-Box (L1) (Q):                   0.08   Jarque-Bera (JB):               562.26\n",
       "Prob(Q):                              0.77   Prob(JB):                         0.00\n",
       "Heteroskedasticity (H):              16.01   Skew:                            -0.81\n",
       "Prob(H) (two-sided):                  0.00   Kurtosis:                         9.80\n",
       "===================================================================================\n",
       "\n",
       "Warnings:\n",
       "[1] Covariance matrix calculated using the outer product of gradients (complex-step).\n",
       "\"\"\""
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# For non-seasonal data\n",
    "#p=1, d=1, q=0 or 1\n",
    "from statsmodels.tsa.arima.model import ARIMA\n",
    "model=ARIMA(df['Cost'],order=(2,1,3))\n",
    "model_fit=model.fit()\n",
    "model_fit.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Date'>"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df['forecast']=model_fit.predict(start=190,end=278,dynamic=True)\n",
    "df[['Cost','forecast']].plot(figsize=(12,8))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Vishi Ved\\AppData\\Roaming\\Python\\Python310\\site-packages\\statsmodels\\tsa\\base\\tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.\n",
      "  self._init_dates(dates, freq)\n",
      "C:\\Users\\Vishi Ved\\AppData\\Roaming\\Python\\Python310\\site-packages\\statsmodels\\tsa\\base\\tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.\n",
      "  self._init_dates(dates, freq)\n",
      "C:\\Users\\Vishi Ved\\AppData\\Roaming\\Python\\Python310\\site-packages\\statsmodels\\tsa\\statespace\\sarimax.py:1009: UserWarning: Non-invertible starting seasonal moving average Using zeros as starting parameters.\n",
      "  warn('Non-invertible starting seasonal moving average'\n",
      "C:\\Users\\Vishi Ved\\AppData\\Roaming\\Python\\Python310\\site-packages\\statsmodels\\base\\model.py:604: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals\n",
      "  warnings.warn(\"Maximum Likelihood optimization failed to \"\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Date'>"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAsYAAAHgCAYAAACmdasDAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAAsTAAALEwEAmpwYAACOYUlEQVR4nOzdd3xbd/X/8dfVsOS945nE2XsnTfcetJTuQim0jEKZP/b88oUvfIEvm0LZmxYKLXTTvVe6Mpq9t/e25SHJGvf3hyTHjveQJVvv5+ORR+Kre3U/spP26Oh8zjFM00REREREJNFZYr0AEREREZF4oMBYRERERAQFxiIiIiIigAJjERERERFAgbGIiIiICKDAWEREREQEAFusFwCQl5dnlpWVxXoZIiIiIjLFbd68ucE0zfz+HouLwLisrIxNmzbFehkiIiIiMsUZhnFsoMdUSiEiIiIiggJjERERERFAgbGIiIiICBAnNcb98fl8VFRU4PF4Yr2UScHpdFJaWordbo/1UkREREQmpbgNjCsqKkhPT6esrAzDMGK9nLhmmiaNjY1UVFQwa9asWC9HREREZFKK21IKj8dDbm6uguJhMAyD3NxcZddFRERExiBuA2NAQfEI6HslIiIiMjZxHRjHg5qaGm644QbmzJnDmjVruOyyy9i/f/+InuP//u//orQ6ERERERkvCowHYZomV199Neeeey6HDh1i8+bNfO9736O2tnZEz6PAWERERCT+KTAexPPPP4/dbuejH/1o97EVK1Zw5pln8sUvfpGlS5eybNky7rnnHgCqq6s5++yzWblyJUuXLuXll1/mK1/5Cm63m5UrV/Ke97wnVi9FRERERIYQt10pevrWf3axu8o1rs+5uDiD/3nHkkHP2blzJ2vWrOlz/P7772fr1q1s27aNhoYG1q1bx9lnn80//vEPLrnkEr72ta8RCATo7OzkrLPO4pe//CVbt24d1/WLiIiIyPiaFIFxvHnllVd497vfjdVqpaCggHPOOYeNGzeybt06PvjBD+Lz+bjqqqtYuXJlrJcqIiIiIsM0KQLjoTK70bJkyRLuvffeYZ9/9tln89JLL/Hoo4/y/ve/n8997nPcfPPNUVyhiIiIiIwX1RgP4vzzz8fr9fL73/+++9j27dvJysrinnvuIRAIUF9fz0svvcQpp5zCsWPHKCgo4MMf/jAf+tCH2LJlCwB2ux2fzxerlyEiIiIiwzApMsaxYhgGDzzwAJ/5zGf4wQ9+gNPppKysjJ/97Ge0t7ezYsUKDMPghz/8IYWFhdxxxx386Ec/wm63k5aWxp133gnArbfeyvLly1m9ejV33XVXjF+ViIiIiPTHME0z1mtg7dq15qZNm3od27NnD4sWLYrRiiYnfc9EREREBmcYxmbTNNf295hKKUREREREUGAsIiIikjD8gSBn/uA5HtpaGeulxCUFxiIiIiIJotXto6LZzY6K1lgvJS4pMBYRERFJEC6PH4CGdm+MVxKfFBiLiIiIJAiXO9Q+tl6Bcb8UGIuIiIgkCJcnFBg3tHXFeCXxSYHxIG6//XYWLVrEe97znlgvhQcffJDdu3fHehkiIiIyibncoVIKZYz7p8B4EL/+9a95+umnhzWUw+/3R3UtCoxFRERkrCIZ4+bOLnyBYIxXE38UGA/gox/9KIcPH+bSSy/lJz/5CVdddRXLly/n1FNPZfv27QB885vf5KabbuKMM87gpptuor6+nmuvvZZ169axbt06NmzYAEB7ezsf+MAHWLZsGcuXL+e+++4D4GMf+xhr165lyZIl/M///E/3vb/yla+wePFili9fzhe+8AVeffVVHn74Yb74xS+ycuVKDh06NPHfEBEREZn0IjXGpglNHSqnONnkGAn9+FegZsf4PmfhMrj0+wM+/Nvf/pYnnniC559/nm9961usWrWKBx98kOeee46bb76ZrVu3ArB7925eeeUVkpOTufHGG/nsZz/LmWeeyfHjx7nkkkvYs2cP3/72t8nMzGTHjtBraG5uBuC73/0uOTk5BAIBLrjgArZv305JSQkPPPAAe/fuxTAMWlpayMrK4oorruDyyy/nuuuuG9/vg4iIiCSMSMYYoL7NS0GGM4ariT+TIzCOsVdeeaU7y3v++efT2NiIy+UC4IorriA5ORmAZ555ple5g8vlor29nWeeeYa77767+3h2djYA//rXv/j973+P3++nurqa3bt3s3jxYpxOJ7fccguXX345l19++US9TBEREZniIjXGoDrj/kyOwHiQzG6spaamdv85GAzy+uuv43QO/e7ryJEj/PjHP2bjxo1kZ2fz/ve/H4/Hg81m48033+TZZ5/l3nvv5Ze//CXPPfdcNF+CiIiIJAiXx4fdauALmNS3KTA+mWqMh+Gs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",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import statsmodels.api as sm\n",
    "model=sm.tsa.statespace.SARIMAX(df['Cost'],order=(2,1,3), seasonal_order=(2,1,3,6))\n",
    "results=model.fit()\n",
    "df['forecast']=results.predict(start=190,end=257,dynamic=True)\n",
    "df[['Cost','forecast']].plot(figsize=(12,8))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pandas.tseries.offsets import DateOffset\n",
    "future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Cost</th>\n",
       "      <th>Seasonal First Difference</th>\n",
       "      <th>forecast</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2025-02-01</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2025-03-01</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2025-04-01</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2025-05-01</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2025-06-01</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Cost Seasonal First Difference forecast\n",
       "2025-02-01  NaN                       NaN      NaN\n",
       "2025-03-01  NaN                       NaN      NaN\n",
       "2025-04-01  NaN                       NaN      NaN\n",
       "2025-05-01  NaN                       NaN      NaN\n",
       "2025-06-01  NaN                       NaN      NaN"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "future_datest_df.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "future_df=pd.concat([df,future_datest_df])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Vishi Ved\\AppData\\Roaming\\Python\\Python310\\site-packages\\statsmodels\\tsa\\statespace\\kalman_filter.py:2290: ValueWarning: Dynamic prediction specified to begin during out-of-sample forecasting period, and so has no effect.\n",
      "  warn('Dynamic prediction specified to begin during'\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAsYAAAHSCAYAAADvxw2lAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAAsTAAALEwEAmpwYAACFUklEQVR4nO3dd3ijZ5X///ctWe7dHpexp/de03svhBTaBgKBpQRYWNoCy5bvb2EXdln6stTQNvSEEkhIQhJSSM9kJpne+9jj3uSmfv/+kOSxx72p+fO6rrniefRIumU/8RwdnfscY61FRERERGSmc8R7ASIiIiIiiUCBsYiIiIgICoxFRERERAAFxiIiIiIigAJjERERERFAgbGIiIiICABp8V4AQGlpqZ0/f368lyEiIiIiKW7btm3N1tpZQ92WEIHx/Pnz2bp1a7yXISIiIiIpzhhzYrjbVEohIiIiIoICYxERERERQIGxiIiIiAiQIDXGIiIiIjIyv99PTU0NHo8n3ktJCpmZmVRXV+NyucZ8HwXGIiIiIkmgpqaGvLw85s+fjzEm3stJaNZaWlpaqKmpYcGCBWO+n0opRERERJKAx+OhpKREQfEYGGMoKSkZd3ZdgbGIiIhIklBQPHYT+V4pMBYRERGRMauvr+f2229n0aJFbNq0iRtvvJGDBw+O6zH+8z//c5pWNzkKjEVERERkTKy13HbbbVx++eUcOXKEbdu28V//9V80NDSM63EUGIuIiIhIUnvqqadwuVx84AMf6Du2bt06Lr74Yj71qU+xevVq1qxZw7333gtAXV0dl156KevXr2f16tU8++yzfOYzn6G3t5f169dzxx13xOulDEldKURERESSzOce3MPe0+4pfcyVs/P5t9evGvGc3bt3s2nTpkHHf//737N9+3Z27NhBc3Mz55xzDpdeeim//OUvue666/iXf/kXgsEgPT09XHLJJXzrW99i+/btU7r+qaDAWEREREQm5bnnnuOtb30rTqeT8vJyLrvsMl555RXOOecc3v3ud+P3+7n11ltZv359vJc6IgXGIiIiIklmtMzudFm1ahW//e1vx3z+pZdeyjPPPMNDDz3Eu971Lj7xiU9w5513TuMKJ0c1xiIiIiIyJldeeSVer5e7776779jOnTspLCzk3nvvJRgM0tTUxDPPPMO5557LiRMnKC8v533vex/v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",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "future_df['forecast'] = results.predict(start = 279, end = 1000, dynamic= True)  \n",
    "future_df[['Cost', 'forecast']].plot(figsize=(12, 8))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
