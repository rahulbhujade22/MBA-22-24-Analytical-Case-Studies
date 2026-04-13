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
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.read_excel(r'C:\\Users\\Vishi Ved\\Downloads\\JITINFRA.xlsx')"
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
       "      <th>0</th>\n",
       "      <td>2017-Mar</td>\n",
       "      <td>0.882703</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2017-Apr</td>\n",
       "      <td>1.037092</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2017-May</td>\n",
       "      <td>0.967032</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2017-Jun</td>\n",
       "      <td>0.763865</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2017-Jul</td>\n",
       "      <td>0.732184</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Date  Stock Price\n",
       "0  2017-Mar     0.882703\n",
       "1  2017-Apr     1.037092\n",
       "2  2017-May     0.967032\n",
       "3  2017-Jun     0.763865\n",
       "4  2017-Jul     0.732184"
      ]
     },
     "execution_count": 5,
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
       "      <th>Stock Price</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>72</th>\n",
       "      <td>2023-Mar</td>\n",
       "      <td>1.136148</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>73</th>\n",
       "      <td>2023-Apr</td>\n",
       "      <td>1.320778</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>74</th>\n",
       "      <td>2023-May</td>\n",
       "      <td>1.856607</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75</th>\n",
       "      <td>2023-Jun</td>\n",
       "      <td>4.131213</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>76</th>\n",
       "      <td>2023-Jul</td>\n",
       "      <td>7.303800</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        Date  Stock Price\n",
       "72  2023-Mar     1.136148\n",
       "73  2023-Apr     1.320778\n",
       "74  2023-May     1.856607\n",
       "75  2023-Jun     4.131213\n",
       "76  2023-Jul     7.303800"
      ]
     },
     "execution_count": 6,
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
   "execution_count": 7,
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
       "      <td>2017-Mar</td>\n",
       "      <td>0.882703</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2017-Apr</td>\n",
       "      <td>1.037092</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2017-May</td>\n",
       "      <td>0.967032</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2017-Jun</td>\n",
       "      <td>0.763865</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2017-Jul</td>\n",
       "      <td>0.732184</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Date      Cost\n",
       "0  2017-Mar  0.882703\n",
       "1  2017-Apr  1.037092\n",
       "2  2017-May  0.967032\n",
       "3  2017-Jun  0.763865\n",
       "4  2017-Jul  0.732184"
      ]
     },
     "execution_count": 7,
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
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# For dropping rows\n",
    "#df.drop(106,axis=0,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Vishi Ved\\AppData\\Local\\Temp\\ipykernel_11120\\2147723538.py:2: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.\n",
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
   "execution_count": 10,
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
       "      <td>2017-03-01</td>\n",
       "      <td>0.882703</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2017-04-01</td>\n",
       "      <td>1.037092</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2017-05-01</td>\n",
       "      <td>0.967032</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2017-06-01</td>\n",
       "      <td>0.763865</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2017-07-01</td>\n",
       "      <td>0.732184</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>72</th>\n",
       "      <td>2023-03-01</td>\n",
       "      <td>1.136148</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>73</th>\n",
       "      <td>2023-04-01</td>\n",
       "      <td>1.320778</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>74</th>\n",
       "      <td>2023-05-01</td>\n",
       "      <td>1.856607</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75</th>\n",
       "      <td>2023-06-01</td>\n",
       "      <td>4.131213</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>76</th>\n",
       "      <td>2023-07-01</td>\n",
       "      <td>7.303800</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>77 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "         Date      Cost\n",
       "0  2017-03-01  0.882703\n",
       "1  2017-04-01  1.037092\n",
       "2  2017-05-01  0.967032\n",
       "3  2017-06-01  0.763865\n",
       "4  2017-07-01  0.732184\n",
       "..        ...       ...\n",
       "72 2023-03-01  1.136148\n",
       "73 2023-04-01  1.320778\n",
       "74 2023-05-01  1.856607\n",
       "75 2023-06-01  4.131213\n",
       "76 2023-07-01  7.303800\n",
       "\n",
       "[77 rows x 2 columns]"
      ]
     },
     "execution_count": 10,
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
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.set_index('Date',inplace=True)"
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
       "      <th>2017-03-01</th>\n",
       "      <td>0.882703</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2017-04-01</th>\n",
       "      <td>1.037092</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2017-05-01</th>\n",
       "      <td>0.967032</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2017-06-01</th>\n",
       "      <td>0.763865</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2017-07-01</th>\n",
       "      <td>0.732184</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                Cost\n",
       "Date                \n",
       "2017-03-01  0.882703\n",
       "2017-04-01  1.037092\n",
       "2017-05-01  0.967032\n",
       "2017-06-01  0.763865\n",
       "2017-07-01  0.732184"
      ]
     },
     "execution_count": 12,
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
       "    <tr>\n",
       "      <th>Date</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2023-03-01</th>\n",
       "      <td>1.136148</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2023-04-01</th>\n",
       "      <td>1.320778</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2023-05-01</th>\n",
       "      <td>1.856607</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2023-06-01</th>\n",
       "      <td>4.131213</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2023-07-01</th>\n",
       "      <td>7.303800</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                Cost\n",
       "Date                \n",
       "2023-03-01  1.136148\n",
       "2023-04-01  1.320778\n",
       "2023-05-01  1.856607\n",
       "2023-06-01  4.131213\n",
       "2023-07-01  7.303800"
      ]
     },
     "execution_count": 13,
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
   "execution_count": 14,
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
       "      <td>77.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.853063</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>1.109698</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.052120</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.122665</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.462828</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1.323624</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>7.303800</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Cost\n",
       "count  77.000000\n",
       "mean    0.853063\n",
       "std     1.109698\n",
       "min     0.052120\n",
       "25%     0.122665\n",
       "50%     0.462828\n",
       "75%     1.323624\n",
       "max     7.303800"
      ]
     },
     "execution_count": 14,
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
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Date'>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAWoAAAEGCAYAAABM7t/CAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAAsTAAALEwEAmpwYAAAo2klEQVR4nO3dd3hc1YH38e+Z0aj34irbcsHgbmzZQAymE1oSUkjIm0CABC8hyYbUhWU3jWxCSFnCJgvrF0IgEJYU85JQQgtgCM1yxQ3jbrmpWWVUpp73jxnZslGZqhlJv8/z6NFo5s695yDzm6NzTzHWWkREJH05Ul0AERHpn4JaRCTNKahFRNKcglpEJM0pqEVE0lxGMk5aXl5uq6qqknFqEZFhafXq1Q3W2oreXktKUFdVVVFTU5OMU4uIDEvGmD19vaauDxGRNKegFhFJcwpqEZE0l5Q+ahGRWPh8Pmpra+nq6kp1UZImOzubyspKXC5XxO9RUItI2qitraWgoICqqiqMMakuTsJZa2lsbKS2tpbJkydH/D51fYhI2ujq6qKsrGxYhjSAMYaysrKo/2JQUItIWhmuId0tlvopqEVEUmzj/pZ+X1dQi4ic4NChQ1x11VVMnTqVhQsXcumll7Jt27aozvGjH/0o4mPfUVCLiETOWstHP/pRzjnnHHbs2MHq1av58Y9/zOHDh6M6TzRB3e7x9/u6glpEpIeXXnoJl8vFjTfeePS5efPmceaZZ/Ktb32L2bNnM2fOHB577DEADh48yNKlS5k/fz6zZ8/m1Vdf5ZZbbqGzs5P58+fzmc98ZsBrugcIag3PE5G09P2/bmLzgdaEnnPmuEK++6FZ/R6zceNGFi5c+L7nV6xYwbp161i/fj0NDQ0sWrSIpUuX8vvf/54PfvCD3HbbbQQCATo6OjjrrLP41a9+xbp16yIq10AtagW1iEgEXnvtNT796U/jdDoZPXo0Z599NqtWrWLRokVcf/31+Hw+rrjiCubPnx/1ud2eQL+vDxjUxpiTgcd6PDUF+I619q6oSyMiEqGBWr7JMmvWLP70pz9FfPzSpUtZuXIlTz31FNdeey1f//rXueaaa6K6Ztx91Nbad621862184GFQAfweFSlEBEZIs477zw8Hg/Lly8/+tyGDRsoLi7mscceIxAIUF9fz8qVK1m8eDF79uxh9OjR3HDDDXzhC19gzZo1ALhcLnw+X0TXTHTXx/nADmttn+umiogMZcYYHn/8cW6++WZ+8pOfkJ2dTVVVFXfddRdut5t58+ZhjOHOO+9kzJgxPPjgg/z0pz/F5XKRn5/PQw89BMCyZcuYO3cuCxYs4JFHHun3mgPdTDTW2mgq8BtgjbX2V/0dV11dbbVxgIhEa8uWLcyYMSPVxUi6E+t5+X+9ylP/vHS1tba6t+MjHp5njMkEPgz8sY/XlxljaowxNfX19VEWW0Rk5Gof4GZiNOOoLyHUmu511Le1drm1ttpaW11R0eu2XyIi0ouBuj6iCepPA4/GVRoRkQFE0x07FPVWv4TMTDTG5AEXAitiKZiISCSys7NpbGwctmHdvR51dnb20eeCQUuHN85x1OGTtwNlcZVQRGQAlZWV1NbWMpzvc3Xv8NKt3dt/axo0M1FE0ojL5Ypq55PhYKAbiaBFmUREUmqgG4mgoBYRSamBbiSCglpEJKUU1CIiaU5dHyIiaS6SUR8KahGRFBpoLWpQUIuIpJT6qEVE0ly7x48x/R+joBYRSSG3x09+Zv9zDxXUIiIp1O7xk5eloBYRSVvtngB5Wc5+j1FQi4ikUJvHT75a1CIi6UtdHyIiaU5BLSKS5tzq+hARSW+hFrVuJoqIpK3QqA+1qEVE0pLXH8QbCCZmwosxptgY8ydjzFZjzBZjzBkJKaWIyAjWvc7HQC3qSPdM/CXwN2vtJ4wxmUBuXKUTEZGja1EPdDNxwKA2xhQBS4FrAay1XsAbbwFFREa67rWo87Pj7/qYDNQDDxhj1hpj7jPG5J14kDFmmTGmxhhTM5y3ehcRSZRIuz4iCeoMYAFwj7X2VKAduOXEg6y1y6211dba6oqKiqgLLCIy0nRvGpCfgOF5tUCttfat8M9/IhTcIiISh4S1qK21h4B9xpiTw0+dD2yOs3wiIiNe983EvAGG50U66uMrwCPhER87geviKZyIiBxrUcc96gPAWrsOqI63UCIickwibyaKiEgSuD0BMp0OMjP6j2IFtYhIikSyIBMoqEVEUiaStahBQS0ikjKRrEUNCmoRkZRp96pFLSKS1twRrEUNCmoRkZRp9/gHnD4OCmoRkZRp9/gHnJUICmoRkZRxa9SHiEj6staGuz4U1CIiaanLFyRoB54+DgpqEZGUOLYNl24mioikJXeECzKBglpEJCUiXTkPFNQiIikR6Q7koKAWEUkJtahFRNKcbiaKiKS59vAO5JG0qCPaissYsxtoAwKA31qrbblEROIQTddHpJvbApxrrW2IsUwiItJDpDuQg7o+RERSot3jJzfTidNhBjw20qC2wHPGmNXGmGW9HWCMWWaMqTHG1NTX10dRXBGRkSfSTQMg8qA+01q7ALgE+JIxZumJB1hrl1trq6211RUVFZGXVkRkBHJ7AhGNoYYIg9pauz/8vQ54HFgcc+lERCTiHcghgqA2xuQZYwq6HwMXARvjKqGIyAjnjnDTAIhs1Mdo4HFjTPfxv7fW/i324omISLvHz5jC7IiOHTCorbU7gXnxFkpERI5pj3B3F9DwPBGRlIh0B3JQUIuIpESkO5CDglpEZNAFgpZOn1rUIiJpq90b+VrUoKAWERl00SzIBApqEZFBp6AWEUlz7vBa1LqZKCKSptqjWOIUFNQiIoPOra4PEZH01h7FDuSgoBYRGXS6mSgikuaO3UxUUIuIpKV2jx+HgWxXZBGsoBYRGWTu8Mp54eWjB6SgFhEZZG6PP+JuD1BQi4gMumjWogYFtYjIoHMrqEVE0ls0a1FDFEFtjHEaY9YaY56MqWQiIgJAS6eP4pzMiI+PpkX9VWBL1CUSEZHjtHT6KMp1RXx8REFtjKkELgPui7FcIiICWGtp7vBRnJPgoAbuAr4NBPs6wBizzBhTY4ypqa+vj7gAIiIjSbs3gD9oKU5ki9oYczlQZ61d3d9x1trl1tpqa211RUVFxAUQERlJmju8AAnvo14CfNgYsxv4X+A8Y8zDMZRPRGTEa+7wASS2RW2tvdVaW2mtrQKuAv5urf1sjGUUERnRjgV1ckZ9iIhInJo7w10fUbSoI58aA1hrXwZejuY9IiJyzNEWdRJGfYiISAK0dIaCOuHjqEVEJDGaO7zkZjrJykjCFHIREYnfkSgnu4CCWkRkUDV3+CiKYsQHKKhFRAZVS6dXLWoRkXTW3OGLamgeKKhFRAZVc6cvqskuoKAWERk0oZXzvGpRi4ikqw5vAF/Aqo9aROJnrcVam+piDDvNndEvyAQKahHpxZX3vsHPn9uW6mIMO91LnBZFscQpKKhF5ATWWt7Z38LfNh1KdVGGnZYYljgFBbWInMDt8ePxB9le56bB7Ul1cYaV7q6PEo36EJF41LcdC+dVu5pSWJLh50hH9EucgoJaRE7Q4PYeffyWgjqhupc4LdKoDxGJR3d3R3l+loI6wVo6fWS7HGS7Il85DxTUInKC7qC+ZPYYth5qPXoDTOLX3OGNalPbbgpqETlOQ5sHY+Di2WOwFlbtVqs6UWJZ5wMiCGpjTLYx5m1jzHpjzCZjzPdjKqGIDAn1bi+luZksnFRCptPB2wrqhElaUAMe4Dxr7TxgPnCxMeb0qK8kIkNCg9tDeX4W2S4n8ycU89bOxlQXadho7kxS14cNcYd/dIW/NLdUZJhqcHsoLwiFyeLJpWw80Irb409xqYaHZLaoMcY4jTHrgDrgeWvtW1FfSUSGhO4WNYSCOhC0rNlzJMWlGvqstTR3+qLa1LZbREFtrQ1Ya+cDlcBiY8zsE48xxiwzxtQYY2rq6+ujLoiIpIeGNu/RoF44qQSnw/DWLnV/xKvLF8TrD0Y9KxGiHPVhrW0GXgIu7uW15dbaamttdUVFRdQFEZHUa/f46fQFjgZ1XlYGs8cX8bbGU8ft6KzEKCe7QGSjPiqMMcXhxznAhcDWqK8kImnv2GSXY62+0yeXsn5fC12+QKqKNSw0x7ggE0TWoh4LvGSM2QCsItRH/WTUVxKRtHc0qAuyjj63eHIp3kCQtXubU1Sq4aG5M7YlTgEyBjrAWrsBODXqM4vIkFPfFgqTivxjQV1dVYox8PauJs6YWpaqog15sS5xCpqZKCI9dLeoK3q0qItyXMwYU6gbinGKdYlTUFCLSA/dQV2ad3yYLJ5cypq9R/AFgqko1rCQ7D5qERkhGtweSnJduJzHR8P8CcV0+UKbCUhsmju8ZGVEv3IeKKhFpIeeY6h7mjWuEIDNB1oHu0jDRqyzEkFBLSI99JyV2NOUinyyXQ42KahjFus6H6CgFpEeQut8vD+onQ7DKWMK2XSgJQWlGh6aO2KbPg4KahHpocHtPW6yS0+zxhWy+WAr1mpNtli0dPooUVCLSDw6vQHcHn+vXR8As8YV0dblZ19T5yCXbHg4EuPuLqCgFpGwo2Oo+wjqmeEbiur+iI1uJopI3OqPTh/vvdV3ypgCnA6jG4ox6PIF8PiD6qMWkfg0tB3bfbw32S4nUyvy2HxQQR2to5Nd1PUhIvFocIfW+egrqCHUT62uj+h1L8ikm4kiEpfuPuqyPkZ9QGjkx+FWz9FjJTJH2kMtanV9iEhcGtweCrMzyMroe4rzsRuK6v6IRktn96YB6voQkTj0Ndmlp1ljiwCN/IhWPAsygYJaRMIa2rx9Ds3rVpTrorIkRy3qKHUvcaqgFpG4RNKiBpg5tlCLM0WpucNHZoaDnBhWzgMFtYiE1bs9A7aoITTyY3djO26PfxBKNTy0dHopznFhjInp/QpqEaHLF6Cty9/nOh89zRpXiLWwVeOpI3akPfZZiRDZLuQTjDEvGWM2G2M2GWO+GvPVRCQtNbYPPIa626zxGvkRrXiWOIXIWtR+4BvW2pnA6cCXjDEzY76iiKSdgWYl9jSmMJvSvMyIRn7UtXbFXbaBBIOWLz68mp/8bWvSrxWreJY4hQiC2lp70Fq7Jvy4DdgCjI/5iiKSdhqOrvMxcFAbY5g1rnDAFvVzmw6x+Ecv8vqOhoSUsS8r1u7nmY2HuOflHaza3ZTUa8UqniVOIco+amNMFXAq8FYvry0zxtQYY2rq6+tjLpCIDL6jQR1BHzWEJr5sO9yG19/7ZrfWWu564T0AHvjH7oSUsTetXT7ueGYr8yYUM744h9sefyctN+ANrZyX3K4PAIwx+cCfgZutte/7KLXWLrfWVltrqysqKmIukIgMvkjW+ehp5thCfAHLe3Vtvb7+4pY6Nh9sZfrofF7ccpj9zclZw/ruF96jsd3DDz8ym+9/eBbbDru5/7VdSblWrLp8ATp9AYpyktyiNsa4CIX0I9baFTFfTUTSUn2bh4KsjIh3yJ41LjRD8fXtje97zVrL3X9/j4mludx3zSIAHnlzT8RlibRFvL2ujd++vpurFk1gTmURF8wczYUzR3PXC9vY19QR8fWSrSXOyS4Q2agPA9wPbLHW/iLmK4lI2op0sku3qRV5fGBqGT9//l22nDBM75Vt9WyobeGmc6YysSyX82eM5rFV+/D4AwOe90BzJ6f/6MUBW8XWWr73l83kZjr55kUnH33+ex+ehcHwvb9sSpstw+rDN2qTPepjCXA1cJ4xZl3469KYrygiaSe0+3jkQWKM4ZdXnUphtoubHllDa1eo1Wit5e4X32N8cQ4fW1AJwDVnTKKx3cvT7xwc8Lw/ffZdGtu9/Py5dzncz4iRZzcd4rXtDXzjopMp69FdM744h69deBIvbq3juc2HI65PMnXfTJ0/sTjmc0Qy6uM1a62x1s611s4Pfz0d8xVFJO2ENrWNvEUNUFGQxa/+zwL2NnXwL3/agLWW13c0smZvMzeePYXMjFC8LJlazpTyPB56o//ujw21zTy+dj8fmT8Of8By59/e7fW4Tm+A25/cwiljCvjMaRPf9/p1SyZzypgCvveXTXR4Uz978oUtdcwYW8j44pyYz6GZiSJCfZsn6qAGWDy5lH+5+GSe2XiI+1/bxd0vvsfowiyurJ5w9BiHw/DZ0yexdm8zG/f3PvbaWssPn9xCeX4mP7xiNtedWcWf19Syfl/zcccFgpabH1vLgZZOvv/hWWQ43x9hLqeD26+YzcGWLu59ZWfUdUqkI+1eanY3ccGMUXGdR0EtMsJ5/UFaOn0xBTXADWdN4aKZo/nR01t4a1cT/7R06vtuSn58YSU5LicPvbG713M8u+kwb+9u4uYLplOQ7eLL506jPD+THzy5+bi+5h8/vYVnNx3m3y+byWlTyvos06KqUi6fO5b/eWVH0kacROLlbXUELZw/Y3Rc51FQi4xwje39b2o7EGMMP71yHhNKc6koyOLTi9/fHVGU4+KKU8fzxLoDNHd4j3vN6w9yxzNbOGlUPlctCrXEC7JdfPOik1m95wh/3RDq237w9d3c99ourltSxfVnTh6wXLdccgoWuDOFMxZf2FJHRUEWc8cXxXUeBbXICNfQFt0Y6t4U5bh44ktLeOJLS8jJ7H2I3zVnTMLjD/KDJzez7XDb0Zby797cw+7GDv71shnHdWVcWT2BmWMLuePpLfx1/QG+/9dNXDhzNP92WWQrWFSW5LLsrCk8se4Aq/cciblusfL6g6x8t57zTxmFwxHbqnndFNQiI1xdW2h0RUUUw/N6U5ybybh+bpjNGFvIJ6sreXztfi76z5Wc//NXuOOZrdz94nucdVI550w/fqKc02H47odmcqCli688upY544u4+6pTcUYRel88ZyqjCrK4/cnNBIODO1zv7V1NtHn8cXd7gIJaZMTb1dAOwOSyvKRf685PzOOtW8/n9itmM644h//76k7cHj//eumMXtdqPm1KGR9fUMmU8jzu+9yiPlvrfcnLyuDbF5/Cun3N/GX9gURVIyIvbDlMVoaDM6eVx32ujASUR0SGsO11bsryMinJi31CRjRGFWZz9emTuPr0STR3eGlwe5k2Kr/P43925VwCQdvrCI9IfOzU8Tz0xm7ueGYrcyuL8AUsHV4/nb4AE0pymVCaG2tV+mSt5cWth1kyrTzqD5feKKhFRrjtdW6m9hOUyVScmzngYkXGGDKcsffxOhyG71w+k0/c+wbn/fyV417LynCw/Jpqzp6e2PWJ3qtzs6+pkxvPnpqQ8ymoRUYway3b691cOmdsqouSVNVVpTxw7SIa3B5yMzPIzXSSmeHgP57awg0P1vDrzyzgwpnx9yV3ez48K/L8UxJzTgW1yAjW2O6lucPH1IrUtKgH07mnvH/SyaM3nM41D7zNFx9ezS+vOpXL5ibmA+vFLYeZM76IMUXZCTmfbiaKjGDb69wA/fYRD2dFuS4e/vxi5k8o5iuPrmHFmtq4z9ng9rB2XzPnxzkbsSe1qEVGsB31IzuoITS55sHrF/OFB2v4+h/W88OntjC+OCf0VZLDpXPGsHBSacTne2lrHdbCBQkYltdNLWqREWx7nZvcTCfjEvQn+lCVl5XBA9ct4t8um8EHZ42hJC+T9+raeOStPVy1/E2e2jDwyn8Ah1q6+MXz25hUlsuscYUJK59a1CIj2PY6N1Mr8nsdwzzSZLucfOGsKcc919Lp4wsPruIrj66hrWsOV/UyPb6b2+Pnut+uorXTxx9uPCOh/03VohYZwXbUuUd0t8dAinJcPHT9aSydXsEtK97h3ld29HqcLxDkpkfWsO1wG//92YVHd8BJFAW1yAjV7vFzoKVLQT2AnEwny6+u5kPzxnHHM1v5wV83s6PefXStEmsttz3+Diu31fPjj85J+JhsUNeHyIjVfSNxJAzNi1dmhoO7PjWf4hwXv/nHLn7zj12U5LpYOKmEvKwMnlh3gH8+/yQ+uWjCwCeLgYJaZIQ6NjQv+Wt8DAdOh+H2K2Zz7ZIqanY3UbP7CKv3HGFnQzufqp7A1y44KWnXHjCojTG/AS4H6qy1s5NWEhEZVNvr3GQ4DJMGYTGm4WRqRT5TK/L51KLQjcVObyAh63n0J5I+6t8CF8dycmst9726k589+y4HW1K3y4KIvN+OejeTynJxxbjYkYQkO6Qhgha1tXalMaYq2hNba/nBk5t54B+7AbjnlR1cPHsM1y+pYsHEEg0HEkmx7RrxMWQk7KPUGLPMGFNjjKmpr6/n1hXv8MA/dnPdkipe/fa5fP7Myby6rZ6P3/MGH7/ndRrcnkRdWkSi5AsE2dPYoaAeIhIW1Nba5dbaamttdZczj/9dtY8vnzuN71w+kwmlufzrpTN449bzuf0js9h0oJUv/34N/kAwUZcXkSjsaWzHH7QK6iEiKZ1TzZ0+vvXBk/nmB08+rosjLyuDq8+o4scfm8ObO5v48TOp23RSZCQ7OuKjoiDFJZFIJGV43tiibL507rQ+X//Ygko21LZw/2u7mFtZxEfmj09GMUSkD91BPaVCIz6GggFb1MaYR4E3gJONMbXGmM8P9J5IdjO+7bIZLK4q5V/+vIFNB1oiKqyIJMb2OjfjirLJy9JUiqEgklEfn07GhV1OB7/+zAI+9F+v8U+/W80jXziNiaW5SRsN0ukNsKepnT2NHexr6mBvU+h7Y7uXz51RxccXVibluiLpaEd9e8q235LopfTjtKIgi3uvXsgn732Ds3/6MmV5mcwaX8TscYXMm1DMGVPLKMx2xXRuXyDIf/19O2/tbGRPYweHWruOe70gK4OJZbkEgpZv/HE9a/cd4d8vn0lWRvLHRIqkUjBo2VHv5lNJmu4siZfyv3vmTyjmmZvP4vXtDbyzv4WN+1tZvnIn/qDF6TAsmFjM0pMqOGt6BaeMKSDbNXCQtnb5uOnhNby2vYH5E4r5wLQyqsryqCrPY1JpLpPKcinKcWGMwR8I8tPn3uV/XtnJxv2t3PPZBYwtyhmEmoukxsHWLjq8AY34GEJSHtRwbEpmN48/wLq9zax8r56V2xr4+fPb+Pnz2zAGxhZmM7kij6qyPBZMLOGSOWPIzTxWjdojHVz/21XsrG/nzk/M5ZPV/bcaMpwObr1kBvMri/nmH9dz+d2v8bMr53HOyRWalCPD0rERHwrqocJ0L9WXSNXV1bampiZh52t0e3hzZxPb69zsbmxnV0M7O+vdtHb5yct0ctncsVxZPYGsDAeff7CGLm+Ae69eyJJp5VFdZ3udmxsfXs32OjcLJhbzz+efxNnTFdgyvNz/2i5uf3Izq//tAsoiuPEvg8MYs9paW93ba2nRoh5IWX7W+3YHttayavcR/lizjyc3HOQPNaFNKccX5/DITacxfXT040Onjcrnya+cyR9X13LPS9u59oFVzK0s4qZzpnHG1DKKcmLrLxdJJ9vr3JTkuhTSQ8iQaFEPpN3j55mNh3j3UCs3LJ3CqIL493/z+oOsWFPLr1/ezr6m0IJSE0pzmDW2iFnjCrlkzhimjdJkARlaOr0BzrrzJeaML+SB6xanujjSQ38t6mER1MnkCwR5fUcjG/e3sPlAK5sOtLC7sQOHgU8srOTmC6Yzrlg3H2VouO/VnfzwqS384Z/OYPHkyHfWluQb8l0fqeRyOjh7esVx2+s0uj3c8/IOHnpjD0+sO8C1S6q46expFOWqa0TSV4fXzz0v7+DMaeUK6SFGC9HGoCw/i3+7fCZ//+bZXDZ3LMtX7uTsn73EijW1JOMvFJFE+N0be2hs9/K1C5O3E4kkh4I6DpUlufzik/N56itnMaU8j6//YT3X/XYVB5q1SYKkF7fHz72v7GDp9AoWTlJreqhRUCfAzHGF/PHGD/Cdy2fy1s4mLvrPlTz85h4t4ypp48HXd3Okw5fUff0keXQzMcH2NXVw64p3eG17AyW5Ls47ZTQXzhzN0unlRyfmtHv8HG7torXLz6TSXEryMlNcahnO2rp8nHXnSyyYWMJvrl2U6uJIH3QzcRBNKM3ld59fzPObD/PMxkM8v/kQf15TS1aGg3HFOdS1dtHuDRz3nlEFWZw8poDpowtYOKmEM08qj3mNE5ET/fYfu2nu8HGzWtNDllrUSeYLBFm1q4nnNh+m3u1hdEE2owqzGF2YRX6Wi90N7Ww91Ma2w6Evjz9IhsOwYFIJ55xcwfwJxXT5ArR2+mnt8tHW5aciP4uJZblUleUxqiALh8Mcdz1/wA7KhpuS3rz+IP9v7X5uf2ozp00u477P9dpYkzShFnUKuZwOPjCtnA9EMJ3dHwiydl8zL22t4+V367nzb+8O+J6sDAdFOS46fQG6fAF8gdAHb0mui6ry0Jook8KrBNYe6WRfUwe1Rzpp6fSFPzCyGVOYzdiibGaPL+K0KaUJmTAkqdPh9fPo2/u479WdHGzpYta4Qm67bEaqiyVxUIs6jR1u7eK9w27yszMozM6gMMdFXmYG9W0e9jS1s7uxgz0N7bg9fnIyneS4Ql8Oh2F/cye7G9rZ3dDOgZYuHAbGFuVQWZJDZUkuxbku6ts8HGrp4lBrF4dauvCGb35OqcjjtMllLJ5cwoKJJUldJ1wSwx8Ismr3EZ7ddIgn1u3nSIeP0yaXctO501h6Url+f0OAZiaOcF2+AE6HweXse5CPPxBk04FW3tzZyFu7mli1q4k2jx+A0rxMTp1QzPwJxUwfU8C0UflMKs0lo5/zyfG8/iBN7V5au3y0dvpo6fQRtDCxNJeJpblRd1VZaznY0sWG2mb+vrWOF7bU0dTuJTPDwXknj+KGpVNYOKkkSbWRZFBQS9QCQcu2w22s3dvM2r1HWLuv+ejymACZTgdV5blMrcinqjyPyeV5TCnPY3RhNh5/gHZPgHavn05vgKwMJ4U5GRTluCjMdpGfndHvh0a3urYuVu06wtu7Gnn3cBsGQ4bT4HQYMhwO5lYWcemcsYO2rrK1Fn/Q4vEHae300dzho7nTS0uHD2OgODeTktxMSnJDa52v29dMzZ4manYf4Z3alqN/sfRmVEEWE0tD66RnZzrJdTnJyXSSleEgw+kgwxGqty8QZPOBVt7Z30KD2wuENsE495RRXDx7DGdPr9D2WkNU3EFtjLkY+CXgBO6z1t7R3/EK6uHJ7fGzo87Ne3Vutte52V7Xxs76dvY2deAPRveB73QYsjMcZLuc4S8HuZkZ5GQ6yc10srexg50N7QDkuJzMGFuA02EIBC2BoKXTF2Db4dAHxyljCrh0zliWTCunONdFQVYG+dkZ5Lic7/uT31pLU7uXneGlcvc2dXCkw0dzhzcUvB0+uvwBfIEgPr/FFwjiDQTx+kPfo23XuJyGOeOLqK4qZVJZLoXZLopyQl8W2NvUwd7GUDfWvqYO3B4/nb4And4Anb4AHl8QfzB49N6Dw8BJowqYPb6IuZVFzB5fxOzxhdqZaBiIK6iNMU5gG3AhUAusAj5trd3c13sU1COLPxCk9kgnuxrbqW/1kJ3pJC/TSV5WKCw9/iAtncf+5Hd7/Hj8Abp8Qbp8oe+dPj8d3mMBVZGfxWlTSlk8uYxZ4wp7bYEfaunimY0Hefqdg9TsOfK+EO3+MMhyhVqm2S4nR8KB3C3DYSjODQVncW4mxeEWbabTgcsZ6i5yOR1kZTjIzHCQ6XSQ5XJQmO0Kvy8zHLqW5g4fRzq8HOnw4fEFmFtZzNzKooh2JYpEIGix1qrLaZiKd9THYmC7tXZn+GT/C3wE6DOoZWTJcDpCI0zK8wb1umOKsrluyWSuWzKZw61dbD7QenQIY1uXH7fHh8cXpMsfCH8PUpCdwdSKfKZU5DG1PJ/xJTk4HUPjRluonEOjrJJYkQT1eGBfj59rgdNOPMgYswxYBjBx4sSEFE4kUqMLsxldqGGFMjwl7G8oa+1ya221tba6oqJi4DeIiEhEIgnq/UDPHWIrw8+JiMggiCSoVwEnGWMmG2MygauAvyS3WCIi0m3APmprrd8Y82XgWULD835jrd2U9JKJiAgQ4Vof1tqngaeTXBYREemFBmSKiKQ5BbWISJpTUIuIpLmkLMpkjKkH9vR4qghoifI0g/UegHKgYRCuNVjviaU+sV5LdYr9PYP17y7W96lOIYNVp5OstUW9vmKtTfoXsDxd3xN+X026li/G90RdH9VpaNQpjn/jqlOa16m/9wxW18df0/g9sVKdBvc9sRpudYr1OqpTfNcajOv0+Z6kdH0MNcaYGtvHqlVD0XCrD6hOQ4XqlBy6mRiyPNUFSLDhVh9QnYYK1SkJ1KIWEUlzalGLiKQ5BbWISJoblkFtjJlgjHnJGLPZGLPJGPPV8POlxpjnjTHvhb+XhJ8/xRjzhjHGY4z55gnn+lr4HBuNMY8aY1KyOn2C6/TVcH02GWNuTkF1YqnPZ4wxG4wx7xhjXjfGzOtxrouNMe8aY7YbY25JRX2SUKffGGPqjDEbU1WfcDkSUqe+zjPE65RtjHnbGLM+fJ7vJ63QsYxfTPcvYCywIPy4gNCejzOBO4Fbws/fAvwk/HgUsAj4D+CbPc4zHtgF5IR//gNw7RCv02xgI5BLaFGuF4BpQ6A+HwBKwo8vAd4KP3YCO4ApQCawHpg5RH5HvdYp/PNSYAGwMRV1ScLvqdfzDPE6GSA//NgFvAWcnpQyp/IfwSD+Yp4gtDnvu8DYHr+sd0847nu8P6j3AaXhUHsSuCjV9YmzTlcC9/f4+d+Bbw+V+oSfLwH2hx+fATzb47VbgVtTXZ946tTjuapUB3Wi63TieVJdn0TViVDDZw1wWjLKOCy7PnoyxlQBpxL6tBttrT0YfukQMLq/91pr9wM/A/YCB4EWa+1zySttZOKpE6HW9FnGmDJjTC5wKcfv4DPoYqjP54Fnwo9729NzfHJKGrk465SWElWnE86TUvHWyRjjNMasA+qA5621SalTROtRD1XGmHzgz8DN1tpWY47t4GyttcaYfscmhvuoPgJMBpqBPxpjPmutfTh5pe5fvHWy1m4xxvwEeA5oB9YBgeSVuH/R1scYcy6h/1nOHNSCRkF16rtOJ54n6QXvRyLqZK0NAPONMcXA48aY2dbahN9XGLYtamOMi9Av4RFr7Yrw04eNMWPDr48l9CnYnwuAXdbaemutD1hBqL8qJRJUJ6y191trF1prlwJHCPXRDbpo62OMmQvcB3zEWtsYfjqt9vRMUJ3SSqLq1Md5UiLRvydrbTPwEnBxMso7LIPahD4a7we2WGt/0eOlvwCfCz/+HKG+qf7sBU43xuSGz3k+sCXR5Y1EAuuEMWZU+PtE4GPA7xNb2oFFW59wWVcAV1tre36wpM2engmsU9pIVJ36Oc+gS2CdKsItaYwxOYT6ubcmpdCp7shPxhehP00ssIHQn/brCPXFlgEvAu8RGu1QGj5+DKG+zVZCXRy1QGH4te+H/+NvBH4HZA2DOr0KbCY0QuL8IVKf+wi1/ruPrelxrksJ/VWwA7htCP27669OjxK6L+IL/+4+P5Tr1Nd5hnid5gJrw+fZCHwnWWXWFHIRkTQ3LLs+RESGEwW1iEiaU1CLiKQ5BbWISJpTUIuIpDkFtQx5xpiAMWZdeAWz9caYbxhj+v23bYypMsb8n8Eqo0g8FNQyHHRaa+dba2cRmnRwCfDdAd5TBSioZUjQOGoZ8owxbmttfo+fpxCasVgOTCI0USkv/PKXrbWvG2PeBGYQWsb2QeBu4A7gHCAL+LW19n8GrRIi/VBQy5B3YlCHn2sGTgbagKC1tssYcxLwqLW22hhzDqHlXy8PH78MGGWt/aExJgv4B3CltXbXIFZFpFfDevU8EUILuv/KGDOf0CqB0/s47iJgrjHmE+Gfi4CTCLW4RVJKQS3DTrjrI0Bo9bPvAoeBeYTuyXT19TbgK9baZwelkCJR0M1EGVaMMRXAvcCvbKhfrwg4aK0NAlcT2roLQl0iBT3e+izwxfDylxhjphtj8hBJA2pRy3CQE95lwwX4Cd087F6+8r+BPxtjrgH+RmizBAiteBYwxqwHfgv8ktBIkDXhZTDrgSsGp/gi/dPNRBGRNKeuDxGRNKegFhFJcwpqEZE0p6AWEUlzCmoRkTSnoBYRSXMKahGRNPf/AXq0AoYtVcTnAAAAAElFTkSuQmCC",
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
   "execution_count": 16,
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
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_result=adfuller(df['Cost'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
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
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ADF Test Statistic : 0.8374370810537527\n",
      "p-value : 0.9922170584917669\n",
      "#Lags Used : 2\n",
      "Number of Observations Used : 74\n",
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
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Seasonal First Difference']=df['Cost']-df['Cost'].shift(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
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
       "      <th>2017-03-01</th>\n",
       "      <td>0.882703</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2017-04-01</th>\n",
       "      <td>1.037092</td>\n",
       "      <td>0.154390</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2017-05-01</th>\n",
       "      <td>0.967032</td>\n",
       "      <td>-0.070060</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2017-06-01</th>\n",
       "      <td>0.763865</td>\n",
       "      <td>-0.203167</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2017-07-01</th>\n",
       "      <td>0.732184</td>\n",
       "      <td>-0.031681</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2017-08-01</th>\n",
       "      <td>0.687330</td>\n",
       "      <td>-0.044854</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2017-09-01</th>\n",
       "      <td>0.663615</td>\n",
       "      <td>-0.023715</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2017-10-01</th>\n",
       "      <td>0.602458</td>\n",
       "      <td>-0.061157</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2017-11-01</th>\n",
       "      <td>0.741997</td>\n",
       "      <td>0.139539</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2017-12-01</th>\n",
       "      <td>0.765028</td>\n",
       "      <td>0.023030</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2018-01-01</th>\n",
       "      <td>0.786288</td>\n",
       "      <td>0.021260</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2018-02-01</th>\n",
       "      <td>0.660038</td>\n",
       "      <td>-0.126250</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2018-03-01</th>\n",
       "      <td>0.580693</td>\n",
       "      <td>-0.079345</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2018-04-01</th>\n",
       "      <td>0.549872</td>\n",
       "      <td>-0.030821</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                Cost  Seasonal First Difference\n",
       "Date                                           \n",
       "2017-03-01  0.882703                        NaN\n",
       "2017-04-01  1.037092                   0.154390\n",
       "2017-05-01  0.967032                  -0.070060\n",
       "2017-06-01  0.763865                  -0.203167\n",
       "2017-07-01  0.732184                  -0.031681\n",
       "2017-08-01  0.687330                  -0.044854\n",
       "2017-09-01  0.663615                  -0.023715\n",
       "2017-10-01  0.602458                  -0.061157\n",
       "2017-11-01  0.741997                   0.139539\n",
       "2017-12-01  0.765028                   0.023030\n",
       "2018-01-01  0.786288                   0.021260\n",
       "2018-02-01  0.660038                  -0.126250\n",
       "2018-03-01  0.580693                  -0.079345\n",
       "2018-04-01  0.549872                  -0.030821"
      ]
     },
     "execution_count": 21,
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
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ADF Test Statistic : -2.4125743565568745\n",
      "p-value : 0.13821879609545007\n",
      "#Lags Used : 1\n",
      "Number of Observations Used : 74\n",
      "weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary \n"
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
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Date'>"
      ]
     },
     "execution_count": 23,
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
     "image/png": "iVBORw0KGgoAAAANSUhEUgAAAa8AAAA6CAYAAAAHt9ZFAAAQtElEQVR4Ae2dz2vbyBvG93/SyQeDIWDIwafoEkGhIocNgTUU1gRqemgoFBcWk4PpoaRQTKGmsLiH4oXihQUXinsI7peCA8WFgg8BQ8CQgyDwfHlHkj22ZdmRpURpn4VQW5JH73wkzTPvj9H+Bv5HAhsRGKP92IBh6H9ltMcbNcofkwAJkEAogd9C93InCZAACZAACaSQAMUrhReFJpEACZAACYQToHiF8+FeEiABEiCBFBKgeKXwotAkEiABEiCBcAIUr3A+3EsCJEACJJBCAhSvFF4UmkQCJEACJBBOgOIVzod7SYAESIAEUkiA4pXCi0KTSIAESIAEwglQvML5cC8JkAAJkEAKCVC8UnhRaBIJkAAJkEA4AYpXOB/uJQESIAESSCEBilcKLwpNIgESIAESCCdA8Qrnw70kQAIkQAIpJEDxSuFFoUkkQAIkQALhBChe4Xy4lwRIgARIIIUEKF4pvCg0iQRIgARIIJwAxSucD/eSAAmQAAmkkADFK4UXhSaRAAmQAAmEE6B4hfPhXhIgARIggRQSoHil8KLQJBIgARIggXACFK9wPtxLAiRAAiSQQgIUrxReFN2kwRsL1puBvomf1yHgdFAxKug46xzMY34eAgM07llofP95esSeBBOgeAVzScnWEVoPMqiepsScu2TGlxoyfzQxuks209bNCYxaKGar6F1t3hRbSDcBilec10cGzAet+AbMX8Z7cEW69iW+i/GreKy95xkU399diY7bfudjBcbTDn56h1uJdA29+B6ZO9cSxSvOS3ZahRHnbP/sBIU424uzr7G2NULzDyNGD3OM9sNfw2PtHRsovrvD4hWz/f2XhTvNY+3H6ryJolGleK0NjAeGE4hZvMR7KLzsh5/zp9gbs3gpj7WM9vingBPaCYqXjkfyXQWcnOnbftLPFC8k5Hk56L8qwto1kduycXKqOfFOH/U/LZT/Sdls8byL+iMb5o4F654Fa6+C9vCaN34M4uUMe2gel2Dt5JHLZpDfziO/V0b98zV4XQ3Rfmq7bWyX0dL7cd5G5WAfJ1+0a3LNbsZ/eAzidTXG4L86ygcmzK0cjK088tsmSsctDC7Xt9j5Wkdx11Jt2C96WvjJQf91CdajGMPC65u19MhfXryuHAxPm6j+acHcziGTleueh/2oju41HhkM26jsuW3kH7Yw1HJmo38r2D84QS9Nj8ym4jXyxpntDAzDgLFlovSii5HW76U3XUp2JCJeo/dF5J52ML7qoipgtDyQikkbBvb/1kfU26Ux/HAE08ig+KaP8RXgnNZgZY3rx843Ea+rEbov9pHLWqj+N4RzKdVyrvcw/lx17Vsrt+Gg99yE/XoADJvYNwxknk8j4+LNyc1a/Xy7zGfPvpl4Od+aONrNIHfYQP8CmHisV0M0H2Rg7NTWG3jOWyhuVdAZA92/DBhGES1/AFTenAHj9ybSc+cCv7R4jbo4Ocghs1tFZ+hAjS0P2xhfjdE9NmFki2idz95pgd+cHmo7NurfgOHf+zCMDKb5V/Hm5F6oopumgX0D8VLj25aF8qsW+j/GGF8M0Xt7BGvLUgwCGaVwY/ziddXHScFG4weArycoyOB5PB08JUFrGFFc+zE6f3lekXhG6/4972qz58Ur4HypwRQbtSSva2Pm+gIbWbwc9ORhM0xUfS/1tKpVyznoPDVgFE7Q1x6g0acG6p/80dXr20ULpWwFXQcY/1NSQjXNiUhhhDyI8YbUAu1YRB2yZQPxGjZRzBrIPGh6s2VpS8t3fW/AMgyU/pEYokQESureyWdzMP9sYKDx7L8owH47BNQ9bMDQq9akGMcwYgrjhtsRAmph1ybiNfraRe97eGx15THi8Z520V9HJBas30B8L3uo7hgwdqroeZ517zgzzXd5k43Ci9Vhd3lOMs9knBijdehOWpp+f6QwQibgIooB9t/apsjiNUTjfgEnX2/N8thOnIB4ORhfiH/toPvME6oJKH8Wo62/cfpoPa+hLWJ34//JhZSbdfFiOiEhgtGHo2Dx3MnByOaD9/3RwLLVWs6nihoYTS2/NfEePCYySIkn0Dwfof3Egr1nIRc0mF45GI/FeL9vJbQuvEZ87+He1BbnrIXa83YEb2KFHYHXsof6kklHPmsg54ds546pflw2bHh9zJbRnumjJs7qITdgHPfUzDzniZxzKt7s7KTFGY/hiOftXQ994PM91spH/8YYofOqhsbpMtsCAaiN4iGE2bH4S5d10ITN3DKQ2Q6ezBXfLrvjpn00DG+iuXjSCYewY4ZvbTfs5E2YApoBELf9/thiavmt+XxXz436rFPwdDmGemR+NGCLUB22JkLlR4r0tZajj3XU3vYmxwT3OaatX+rB48luHhkjB3PuWXHvkaqKHgRbMEB9p4Dqx6GadHQ/ddE9m5sAez+80X4GGxu6NX7x8k/nD5S6t+DPYrQwItSM1rwdd9Uf2PQZtm9/lH8jeV6eV2XoCyvnvAcM0fxdxEsTfbgP59KCDs/j0B9El/VsGFF5mTv1pcK6GsMKO1Y3oAa3SNWGfh81rxkzHiuAz1U1uFpv/of2QwOFB03P23K9vUUv1L8e+oTG91i1MKK6lzM4+ve64iWVkOvYsRa46GHD8xZKIny7tYnnsnDGNY5xvpyoEHvucDZPtNDWkg2RPEd/bNEmYZBnWX+OvZD5dcrm/QmK66W7BvtRmGkY0b0XMo9v2ROL7Hk56L+0kdnOq8mvpBDk2i3KV0r6ueS+kc2JiZfMXgWMHjIMmsWoGHOhukZOImrYUE+6z5EQsYkzJBBJvPxBVCt7HbdR1kN73oywIN7DpAvhouHPiKchQzcXJP2deg+uKM62OznBmh/C7VivEZfBtRdje9dP76OUSk8F3Z+hi8c6RueJFHLso+l5+a43OxfCdrqoyD2hD4RBg6USRWl3vR5Oj1rTjukPQj9FGvxDW7zZnZHs9yedWjpi/KE8E9pz7//CNAy/slt+pEK/pgGRIi+Pr99zK5tO4oCI4tV/acI87qrcfqhZaelniJGJiZfMgGWg1C+yP4tRg9TXOux7FiRc5IY99lG/6Qo4f+auPQQTVsMOGv9dMzUfSbwk7i/hVS3UpXsPk6KDaWzftTFcNEbvior/VBCm3oMMuP3XNiwVesggv2vBOqivMYGY0NE+hNuhHRjyMaJ4eYPYVKyknWm+yw0NSh4k6Dp6g1XhCB0/5KgsdPszs17Py3epidiojco9qUZ0K7QkTHP0YXHeGtLZuV3L7Jg7bMnXSIP/krZuY3Mk+696qEpBlZaHmsl3eXlQ05/syTP0ZB92IYf94yqODiuqOjH/RPeeAiaRfqRIhR5HaD+1YElqwA/XPWkHeCwhFNeyI+T3+q5I4tVDLbuPZtDjcD7AUOUOV/fTGQ4wGE2n0bpZN/k5OfHyvQUvYep8a6jE+kzoy7sJdYG7yc5LLF5ev6QnfSGlt59qsKXE/+s1L1BE8YJUuWWNyYxIvAcVYx/30XxsutVUC+NjuGj4eRs/BDL6WFE5HkMPtYhIqjzaJtTD7Viv5YjiBbeyUqrKlDelPFY3tDqSa5jNofR2oHmrU2ucz1UUDBNHH+afZF9MvOKYywEaco/MeKyuvZt5rK4ty+2Y2hr2KdLgH9bgDe+Lar9UNGekwOmzhG2lSMwNu4/PmjjaycD6qzMp+xavzH7dQVOKle6foC+DtBr89apC30v3csRXI3SeSRGVMfNuUTUpXCtStAhyPTsWfxe4JZJ4uYv3zccNdL+PML4YY3w+QPddFfb20TRvLIF8mfwG9VNenCCRiZBcaaC9CWxMTrykZOBDBfaWl1S8b7md1hOoCsTyhHEC/V1s8nKA5lMbuS3TTYzu7qP8qoPBdVMZ0nJU8ZLfeusuzB1Zp2QgJ0n4gzLq/w2WuPirRMNda5ffysPatWDfdx9EPYwrImncb2jFGg56r8soPwz7O0F3hs0qOxaRL26JKl7SkoOhrO/aM2Hu5NU6H3PXQum4id6ykJ7MzLds1JatnZM1P/dzbgHJrg2r4BfLeJYrkSygoheSDNuohnIr4+j9XAHFKjsWQS1siTr4LzR0Sxs2sX/0RdZ3yXU3kTNyKoKw/6iOznwFpRRkXMjSE8OtJpW+qrFnLmd52Uf9Qd599nZt2FLNaEw9ecDLV8oyoAmvazwz69oxaTvkQyTxAiBLcl6VYUvkQNI62/KstNCfdgjB/fRs8Sba81G1EEsT25WYeDmi6prj4pdtq1JkrzsqQeqr+6Vb6ZVYT2+i4U3Ey7dP8x78TcH/hoiGVBxezPKUMnB5ECuf/IvixvN978HRL1bwCZdsDbFjyS8WN28iXtPWJh7rdNPip2EbRwclNDyvevChMbuYVQYYnYUsPZCZpibybu7Wy404s5wXT7hkyyo7lvxsfvMmg/98W7fxPQ77Vb5LL9oJ6ohXGOa/fUONPfo6MEcEzn82ZJAXb05yn+6yE9Wkl/tUkSJ5xq6x+H1i0io7Jgeu+BBVvFY0q3av6ueVg8Gb/Vt/p2Yi4uUXC0wWJ8uNMLcmww3ZeWWpl11U9+ZzOutQTtkxUW9ovRsSytNi+fqu2c/LRMMPf2hVhRdtlGfWQomn565fkbCihK5sPz8we5I1vi2zY42f6odsPHkRAZwrvtDbl89qbVAe5dcdqBLhT01UClqu0S+V1hYnS4J7Zv0dAJW7VVW0sgi6GJxDmD+3/n2VHfqxqz5HFc9V7d7U/hjsl3xX+cOM67BgvVtJ6F1rGW8KGdiv+m5IWYp0JIemLU4WQczIiwv0fKkSHrcKdfh3cXbfwhmDN4TaEfyTpVujTziXNunuWNlPWQ+34llbcYo4diciXm5hhomjf0eAttq98U2b2UhY8X0JuYKN0mEFzbl9cXTuLrYhN7e+piSoD71Xsq5HQiV+0UBR+/8XeYUZWRv1MweYFHxU0NHzZldDtA5zKOyVUHrWvNYrlHybwu3wj7qhf9VsUV9KMH/eMdqP3dyVhDwmf/pSDvXQGjAfu4n4yZtN3szmzdw3sNgoHpZwsiz0OH/6yfc17Jgcyw+rCUgEQV9mEvQL17OX8nDzoIzyXhGVd71pON6byGX26ujLEPXDXfhuPpvmzVSrlz3UdjOwH5QivkpphR1Bpt/GtlX9PDuBqUUibsNEOWci4gUvdpyXRac7JuynzbmY6m11N+3nlTLqGF7RonI2eZUDkHzAXXtnWaSr9L0Oe+O1N16ecNtU7+U09yponoXP6CPZyh/FR2DcwdHuinWKKhQ/W/k8b4DKz2/LCwYkh1bCyfyba+Z/EOX7GnZEafZmfzNA/f6+Nlm+2bPrZ0tGvPQz8DMJkAAJ3CYBVVWrv43jloxJix231P24T0vxipso2yMBEkgNAedzTXu9UmmyQP2mDUyLHTfd7yTPR/FKki7bJgESIAESSIQAxSsRrGyUBEiABEggSQIUryTpsm0SIAESIIFECFC8EsHKRkmABEiABJIkQPFKki7bJgESIAESSIQAxSsRrGyUBEiABEggSQIUryTpsm0SIAESIIFECFC8EsHKRkmABEiABJIkQPFKki7bJgESIAESSIQAxSsRrGyUBEiABEggSQIUryTpsm0SIAESIIFECFC8EsHKRkmABEiABJIkQPFKki7bJgESIAESSIQAxSsRrGyUBEiABEggSQIUryTpsm0SIAESIIFECFC8EsHKRkmABEiABJIk8H/aEub4BVk31QAAAABJRU5ErkJggg=="
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
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAZAAAAEKCAYAAAA8QgPpAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAAsTAAALEwEAmpwYAAAvWklEQVR4nO3deXzV9Z3v8dcnO9nISoCEVUEU0CgBBSyCW2mvey21tVPt1KEz13baenuntr3TxZnOte3cdjpdprXWaqe0yLRa0bG4FbAqraBEBRRkk82wBJIQErJ+7h/nBz1k4+QkJ+eEvJ+Px3mc8/t+f7/ze5/kPPLJb/v+zN0RERHpraR4BxARkcFJBURERKKiAiIiIlFRARERkaiogIiISFRUQEREJCpxLSBm9oCZHTCzDd30m5n9u5ltNbPXzeyisL7bzOzt4HHbwKUWERGI/xbIg8DCHvrfB0wKHouB/wAwswLgq8DFwCzgq2aWH9OkIiJyirgWEHd/HjjcwyzXA7/wkD8BeWY2Cngv8Iy7H3b3I8Az9FyIRESkn6XEO8BplAK7w6b3BG3dtXdiZosJbb0wbNiwGWPGjIlN0j5qb28nKSneG4S9M9gyK29sKW9sxTPvli1bDrl7ccf2RC8gfebu9wH3AVRUVPi6devinKhrq1atYv78+fGO0SuDLbPyxpbyxlY885rZO121J3r53QuEbzKUBW3dtYuIyABJ9AKyHPhYcDbWJUCtu78LPAVcbWb5wcHzq4M2EREZIHHdhWVmvwbmA0VmtofQmVWpAO7+Y+BJ4P3AVqAB+HjQd9jM/glYG7zVPe7e08F4ERHpZ3EtIO7+4dP0O3BnN30PAA/EIpeIiJxeou/CEhGRBKUCIiIiUVEBERGRqKiAiIhIVFRAREQkKiogIiISFRUQERGJigqIiIhERQVERESiogIiIiJRUQEREZGoqICIiEhUVEBERCQqKiAiIhIVFRAREYmKCoiIiERFBURERKKiAiIiIlGJawExs4VmttnMtprZ3V30f9fMKoPHFjOrCetrC+tbPqDBRUQkfvdEN7Nk4IfAVcAeYK2ZLXf3TSfmcffPhc3/aeDCsLdodPfyAYorIiIdxHMLZBaw1d23u3szsBS4vof5Pwz8ekCSiYjIacWzgJQCu8Om9wRtnZjZOGAC8Iew5gwzW2dmfzKzG2KWUkREumTuHp8Vm90MLHT3O4LpvwIudvdPdTHvF4Ayd/90WFupu+81s4mECssV7r6ti2UXA4sBSkpKZixdujQ2H6iP6uvryc7OjneMXhlsmZU3tpQ3tuKZd8GCBa+4e0WnDnePywOYDTwVNv1F4IvdzLsemNPDez0I3Hy6dc6YMcMT1cqVK+MdodcGW2bljS3lja145gXWeRd/U+O5C2stMMnMJphZGnAL0OlsKjObAuQDa8La8s0sPXhdBMwFNnVcVkREYiduZ2G5e6uZfQp4CkgGHnD3jWZ2D6Fqd6KY3AIsDargCecCPzGzdkLHce71sLO3REQk9uJWQADc/UngyQ5tX+kw/bUulnsJmB7TcCIi0iNdiS4iIlFRARERkaiogIiISFSGVAGpb2qNdwQRkTPGkCogNQ0t8Y4gInLGGFIF5HhLW7wjiIicMYZcAWlrj8/QLSIiZ5ohVUAc2Fl9LN4xRETOCEOqgABsrjoa7wgiImeEIVdA3nq3Lt4RRETOCEOqgKSnJPGWtkBERPrFkCogGanJKiAiIv1kyBWQXYcbOKYLCkVE+mxoFZCU0MfdvF9bISIifTW0CkhqMqAzsURE+sOQKiBpKUlkpSWrgIiI9IMhVUAAJo/M4U2dyisi0mdDroBMGZnD5v1HOfUOuSIi0ltDsIDkUtPQwv66pnhHEREZ1OJaQMxsoZltNrOtZnZ3F/23m9lBM6sMHneE9d1mZm8Hj9siXec5I3MAeKtKu7FERPoibgXEzJKBHwLvA84DPmxm53Ux68PuXh487g+WLQC+ClwMzAK+amb5kax3yskCogPpIiJ9Ec8tkFnAVnff7u7NwFLg+giXfS/wjLsfdvcjwDPAwkgWzMtMY2Ruhs7EEhHpo5Q4rrsU2B02vYfQFkVHHzCzecAW4HPuvrubZUu7WomZLQYWA5SUlLBq1SpGpLWwbuu7rFpV0/dP0U/q6+tZtWpVvGP0ymDLrLyxpbyxlYh541lAIvE48Gt3bzKzTwIPAZf35g3c/T7gPoCKigqfP38+axrf5IEXdjD3PfNITU6M8whWrVrF/Pnz4x2jVwZbZuWNLeWNrUTMG8+/nnuBMWHTZUHbSe5e7e4nTpe6H5gR6bI9mTIyh5Y2Z8ch3VxKRCRa8Swga4FJZjbBzNKAW4Dl4TOY2aiwyeuAN4PXTwFXm1l+cPD86qAtIlNG5gLogkIRkT6I2y4sd281s08R+sOfDDzg7hvN7B5gnbsvB/7ezK4DWoHDwO3BsofN7J8IFSGAe9z9cKTrPqs4m5Qk04F0EZE+iOsxEHd/EniyQ9tXwl5/EfhiN8s+ADwQzXrTUpI4qzhbp/KKiPRBYhxBjoNzRubw1rt1GtJERCRKQ7aAXDyxgH21x1m95WC8o4iIDEpDtoDcPKOMMQXDuPf3b9HWrq0QEZHeGrIFJD0lmc9ffQ5vVR3lscqIzwAWEZHAkC0gANeeP5rppcP5f09v4XhLW7zjiIgMKkO6gCQlGXe/bwp7axr5zzXvxDuOiMigMqQLCMDcs4uYN7mYH6zcSm1DS7zjiIgMGkO+gADcvXAKdcdb+NHqrfGOIiIyaKiAAOeNzuXG8lJ+/uJOdlU3xDuOiMigoAISuOvqyaQlJ3Hzj1/i9T018Y4jIpLwVEACZfmZ/ObvZpOWksQHf7yGx1/bF+9IIiIJTQUkzJSRufzuzrlMLx3Op3+9nu8+s4V2XWQoItIlFZAOirLTWfI3F3PzjDK+99zb3Pbzl3nlnYgH+hURGTJOOxqvmc0FvgaMC+Y3wN19YmyjxU96SjLfvvl8po7O5XvPvc0H/mMNFePyWTxvIleeW0JSksU7oohI3EUynPvPgM8BrwBD5nJtM+PjcyfwoZljWLZ2Nz/94w4W/+crjCvMZMbYfCaV5DC5JJvJJTmU5g1TURGRISeSAlLr7r+PeZIBUF1dzYMPPnhK29SpU5k5cyYtLS0sWbKk0zLl5eXcPrecmy4o5usPPE5lbTPPvHGUR9Ynn5wnLzOV6aOySavbzZhhrYzOaCEt2Dk4e/ZszjnnHA4dOsQTTzzR6f3nzZvHxIkTqa+v75QN4IorrmDMmDHs3r2b5557rlP/woULGTlyJNu3b+f555/v1H/NNddQVFTE5s2bWbNmTaf+G2+8keHDh7NhwwbWrVvXqX/RokVkZmZSWVlJZWXlKX01NTXMnTuX1NRU1q5dy8aNGzstf/vttwPw0ksvsWXLllP6UlNTufXWWwFYvXo1O3bsOKU/MzOTRYsWAfDss8+yZ8+eU/pzc3O56aabAFixYgVVVVWn9BcWFnLttdcC8Pjjj7Nt2zZ27tx5sn/kyJEsXLgQgEceeYS6ulPvUFlWVsaVV14JwLJly2hoOPUU7wkTJnDZZZcBsGTJElpaTr0QdfLkycyZMwegy9/t6b57GRkZADQ0NLBs2bJO/RUVFUybNo3a2loeffTRTv2RfveqqqpYsWJFp/7efvdqampO+fnG8rsHcOutt/bpu1daWgoMzHevurr6lP5ovnvhP99Yf/fKy8spLy/v9rt3QiQFZKWZfRt4BDhxf3Lc/dUIlj1jpCQlMT23iem5oR9BY5txsCmZzFFncbAti7U7DrG9OjuY28lNaacwrY2X/QAX7k8lnWa21KeRmdzOsGSnqd2oOp7Cay9Wsffp/WyvgvOzM7m4oIF0HZkSkUHATndDJTNb2UWzu/vlsYkUOxUVFd7Vfzr9paahmfW7anhjby07Dx1jR/Uxdhw6Rk0PQ6RkpiVz7qhcGutr2VTdTlF2GncuOJuPXDyW9JTkbpdLBKtWrWL+/PnxjhEx5Y0t5Y2teOY1s1fcvaJj+2m3QNx9QWwigZktBL5H6J7o97v7vR367wLuIHRP9IPAX7v7O0FfG/BGMOsud78uVjkjlZeZxoIpI1gwZcQp7bUNLRxuaOZIQzO1DS3UNDaTnhIqHOMKMklKMlatWkXuxAv49orNfP3xTdz/xx3c+4HpvGdScZw+jYhIzyI5C2s48FVgXtC0GrjH3Wv7smIzSwZ+CFwF7AHWmtlyd98UNtt6oMLdG8zs74BvAR8K+hrdvbwvGQbK8MxUhmemMoGsHue7aGw+v/qbi3lxazVff3wjf/fLV1n+qblMLM7ucTkRkXiIZG/7A8BRYFHwqAN+3g/rngVsdfft7t4MLAWuD5/B3Ve6+4kjl38CyvphvQnNzLh0UhEP/fUsUpON/7nkVd2rREQSUiTHQCo7/qffVVuvV2x2M7DQ3e8Ipv8KuNjdP9XN/D8Aqtz9n4PpVqCS0O6te939d90stxhYDFBSUjJj6dKlfYkdM/X19WRnn7ql8frBVr77ShPvKUvhr6elxylZ97rKnMiUN7aUN7bimXfBggVdHgPB3Xt8AGuAS8Om5wJrTrdcBO97M6HjHiem/wr4QTfzfpTQFkh6WFtp8DwR2Amcdbp1zpgxwxPVypUru2z/16fe8nFfeML/a93ugQ0Uge4yJyrljS3lja145gXWeRd/UyM5jffvgIeCYyEGHAZuj6qMnWovMCZsuixoO4WZXQl8GbjM3cNPI94bPG83s1XAhcC2nlbY1XUgiaLjOfQn5DuMzxzO3b+pZNvalZRkJM7urO4yJyrljS3lja1EzHvaYyDuXunuFwDnA9Pd/UJ3f60f1r0WmGRmE8wsDbgFWB4+g5ldCPwEuM7dD4S155tZevC6iNBWUfjB9zNGksEHRteRkdTO0r3DOdiU2Kf2isgQ0tVmiQe7jYLnu7p6dLdcbx7A+4EthLYcvhy03UOoYAA8C+wndKyjElgetM8hdArva8HzJyJZ32DchXXCup3VfuE9T/uU//N7f/jlXd7e3j4wwXqgXQCxpbyxpbyRI4pdWCfOOc3pqu5EUas6v4n7k8CTHdq+Evb6ym6WewmY3h8ZBosZ4wr4/Wfew2eXVvIPv32dP249xDdunEZuRuqAZ9m4r5YvPbqBBcWtzB/wtYtIoui2gLj7T4KXz7r7i+F9wQi9MsBKcjP45R0X8+PV2/jOM1tYv+sIn5w3kXmTixlX2PM1Jv3l8LFmFv/iFfbWNPLGHhg5fhe3zBo7IOsWkcQSyXUg34+wTQZAcpJx54KzWfbJ2WSkJvOPj23ksm+v4rJvr+Qrj23glXeOxGzdrW3tfPrXr3Kwvolf/c3FTC1M5u5H3uA7z2w5sUtSRIaQbrdAzGw2oWMNxcGQIifkEhp6ROJoxrh8nvncPHZWN/D8loOs3nKQ/1q3hyV/3sU/XT+Nj1zc/1sF335qMy9ureZbN5/PnLOK+MxF6TxVXcC/P/c279Y08i83TSc1WSNBigwVPR0DSQOyg3nCj4PUEbqGQ+LMzJhQlMWEoixumzOe+qZWPv2rV/nSo2+wt6aBz199Dmb9c5+Sx1/bx0+e385fXTKORRWhs69Tkoxv3Xw+o/KG8e/PvU1Wegpfu25qv6xPRBJfT8dAVgOrzexBDwYwlMSWnZ7CTz9WwT8+toEfrtzG3iONfOvmC0hL6dtWwaZ9dfzDb16nYlw+/3jNeaf0mRl3XTWZt96t45lN+1VARIaQSC4kbAjuBzIVyDjR6INwOPehICU5iX+5cTpl+Zl8+6nN7Ks9zg3lpYwpGMaY/ExG5w0jNdlobGmj/ngrR5taSU9Joiw/s8v3e3bTfj73cCW5w1L40Ucv6rYYXTKxkKc37WdfTSOj84bF8iOKSIKIpIAsAR4GrgH+FriN0NDqkqDMQgfaR+dl8OVHN/DyjsNhfaHhBNo7HPN+37SR3HXVZCaVhPZWtrU7//bsFr7/h61MK83lxx+dwYicDLoza0IBAGt3Hub68tJ+/0wikngiKSCF7v4zM/tM2G6ttbEOJn1344VlXHdBKVV1x9lzuIHdRxrZfbiBtnYnOyOF7PQUcjJS2Hagnp+9sIOnNlZxQ3kpH587gX99ejOrtxzkgzPK+KcbppGR2vN5E+eOyiU7PYWXd6iAiAwVkRSQE7fTe9fM/gewDyiIXSTpT8lJRmneMErzhnFxD/PdPncCP169jYde2skj6/eSmmx848ZpfGTW2IgOxCcnGReNy2ftzsOnnVdEzgyRFJB/DgZS/F+Erv/IBT4X01Qy4Aqy0vjS+8/lE5dOYMmf3uHyc0soH5PXq/eYNT6ff336IEeONZOflRaboCKSMCK5pe0TwctaIGa3t5XEUJKbwV1XnxPVsjPHhzZM171zhKvOK+nPWCKSgHq6kPD79DDmlbv/fUwSyaB1wZg80pKTWLvzsAqIyBDQ0xbIugFLIWeEjNRkzi8bfspZXyJy5urpQsKHwqfNLNP/cn9ykS7NmlDAfc9vp6G5lcy0SA6xichgddpLlM1stpltAt4Kpi8wsx/FPJkMSjMnFNDa7qzfVRPvKCISY5GMcfFvwHuBagAP3Y1wXgwzySA2Y1w+Zmg3lsgQENEgSe6+u0NT4tyYWxJKbkYq547M1fUgIkNAJAVkt5nNAdzMUs3s88CbMc4lg9isCQWs31VDS1t7vKOISAxFUkD+FrgTKAX2AuXBdJ+Z2UIz22xmW83s7i76083s4aD/z2Y2Pqzvi0H7ZjN7b3/kkf4xc3wBjS1tbNhbG+8oIhJobG7j5R2HeXnHYdbvOsKGvbVs2X+U4y3R71Dq8TQZM0sGvufut0a9hp7f+4fAVcAeYK2ZLXf3TWGzfQI44u5nm9ktwDeBD5nZecAthEYIHg08a2aT3V271hLAzAn5QGhgxQvH5sc5jcjQ1dbu/Hl7NY+s38uKDVXUN7V2mic9JYk5ZxUy/5wRLDhnBGMLux6Zuyt2uluRmtkLwOXu3tzb8Kd539nA19z9vcH0FwHc/f+GzfNUMM8aM0sBqoBi4O7wecPn62mdOTk5PmPGjP78GP2mpqaGvLy8eMfolZ4y77ngDlIbD1Gy5XcDmqkng+1nrLyxdSbmdaAtLZfjOaNpyimlIX8Sbek5WGsTWYc3k3n4bay9FU9KBkumPSmFpuzRNOZNoHVYaCSJ5OajJLU2YW3NJLU1k9TezJs/vesVd6/ouL5ITtTfDrxoZsuBYyeDun+nF5+9K6VA+MH5PdBpvL+T87h7q5nVAoVB+586LNvlELBmthhYDJCamkpNTU0fY8dGW1tbwmbrTk+Zk6q3c3zEuRypqcW6H9BgQA22n7HyxlYs87anZdOWO5q23JF4WjbW0oi1NISemxtIOl5DUmMN1t55iyDSvJ6STltWMe1ZRbRlFdGeVUxr3lg8Izc0Q2sTKYd3kLmvktQDb2LtrXS1FZBM6NazbZmFtBZPpi1nFJ6STntKOm3JaXhqbreZIikg24JHEqfe2nZQcPf7gPsAKioqfN26xLzAftWqVcyfPz/eMXqlp8yPVe7lM0sr+fdlT3PZ5OKBDdaNwfYzVt7Y6q+8Ta1tvLGnlpd3HmbtjsO8sbeOQ/VNJ/uz0pI51tz13vWRuRmMKRjGhKIsJpfknHyMyEnnwNEmdlYfY1d1A+8cPkbl5newzDz21x1nf91x6o7/pfgkJxlj8odxwZg8ZozL56Kx+UwZmUNKct/uRnqC/eiOLtsjOQYyORbHQAgdkB8TNl0WtHU1z55gF9ZwQtejRLKsxNH7po3iX3Lf5L7ntyVMARHpq7Z2Z8ehejbuq2Pjvjpe211D5e4amlpDZxyePSKbyyYXM3V0LueNzuXcUbkMH5ZKS1s7dY0t1DS2cPhYM3uONLD7cCO7Djewq7qBP7x1gGXr9pxcT3KS0RZ217eUJGN4GoxNaeWs4mzmnFXIyOHDmFicxVnFWYwtyOrzrauj0WMBcfc2MxtnZmn9fQwEWAtMMrMJhP743wJ8pMM8ywndAXENcDPwB3f3YHfar8zsO4QOok8CXu7nfNIHaSlJfHzuBO79/Vts2FvLtNLh8Y4k0mvNre2s33WEl7ZVs2ZbNa/vreF4S6hYpCUnMWVUDh+9ZBwzxxcwc3w+hdnpXb5PanIShdnpFGanc1bxX0auDldd38SW/fW8feAo79YeZ3TeMMYXZjKuIIvReRm88MfnmT9/bkw/b2/F7RhIcEzjU8BThHbDPeDuG83sHmCduy8Hfgb8p5ltBQ4TKjIE8y0DNgGtwJ06AyvxfOTisfzgD1v5yfPb+f6HL4x3HJGI7Ktp5Nk39/PsmwdYu+MwjS1tmMH00uF8ZNY4po7OZWppLmcVZ5PaT7uIAAqz05mdnc7sswr77T1jLa7HQNz9SeDJDm1fCXt9HPhgN8t+A/hGf+aR/pWbkcqHZ43hgRd38g/vPYcxBZGfHigyUFrb2tm4r47VWw7y9KYqNuytA2BCURaLKsqYc3YRl0woZHhmapyTJp5Ibij1dQAzyw6m62MdSs4cH587gZ+/uJOfvbCDr103Nd5xZIhra3cO1Tex+3ADv9/Rwi92rmXtjsMcbWrFDC4ck8cXFk7hqvNKOHtEdrzjJrzTFhAzmwb8J8F90M3sEPAxd98Y42xyBhidN4zrykfz8NrdfPbKSeRl6la3EhvuzoGjTWw7UM/emkYO1jdx8GjocaCuiX21jVTVHqc17OD0xOJjXFs+mksmFjJ7YiHFOV0fw5CuRbIL6z7gLndfCWBm84GfAnNiF0vOJIvnTeSRV/fyyz+9w6cunxTvODJINTa3cai+iYP1TRyoO86BoDBU1R1n28F6th6o5+jxU6+ryElPoTgnneKcdGaOL2DU8AxG5w1jdF4GdTs3csPC+fH5MGeISApI1oniAeDuq8wsK4aZ5AwzZWQul00u5sGXdnLHeyaSkZoc70iSYJpb26ncXcPre2qoPtbM4frm0POxJqqPNXPoaFOX11IkGRTnpDOhKIvry0czaUQOZ4/Ipix/GCNyMhiW1v13bVWVxoTtq4jOwjKzfyS0Gwvgo4TOzBKJ2CfnTeQj9/+Zf31qM196/7kkJVm8I0kcuTtvVR3lxa2HeGHrIV7ecZiGoECkJBn5WWkUZqVRkJXGBWV5FGWnU5idRnF2+sktihG56RRmpZOs71LcRFJA/hr4OvAIoaFW/hi0iURs9lmFLKoo4/4XdvD2gXr+7UPl5GfpeMhQsr/uOH98+xAvvH2QF7ZWn7xae2JxFjfPKGPOWUXMHJ9PQVYaZioKg0EkZ2EdAf5+ALLIGczM+OYHzuf8sjzueXwT13z/BX5060VcMCYv3tEkBppa29i4r471u0JXaq/fdYQ9RxoBKMxKY+7ZRVw6qYhLzy5idN6wOKeVaEVyFtYzwAfdvSaYzgeWnhhFVyRSZsZHLxnH9NLh/M8lr/LBH6/hc1dN5uYZZTr7ZZCrbWzh9YOtrH3qLdbuOELlnhqag+E9Rg/P4MKx+dw+Zzyzzyrk3JG52oV5hohkF1bRieIBoS0SMxsRu0hyprtgTB5PfPpSPreskm+ueItvPfUWs8YX8L5pI1k4bRQjh2fEO6L0oKG5lS3763ltdw2v7anhtd01bDsYGqQiOWk700qH87FLxjFjXD4Xjs3X7/MMFkkBaTezse6+C8DMxkGCjM8tg1Z+Vho/v30mm/cf5ck3qlix4V2+9vgmvvb4JqaMzOHSs4uYe3YRsyYUkJUeyddU+srdaWptpzYY8C/88U51A1sP1p+8xuKEoux0yscM5/ryUlJqdnH7tZeRmabf11ARyW/6y8ALZrYaMOA9BPfXEOkLM2PKyFymjMzlrqsms/XAUZ7auJ8Xtx7iF2ve4f4XdpCSZJw7KpfJJTmcMzKbySU5nFWcTXFOuk4H7qWm1jY2Vx1lx6ETQ4SHRoI9cPQ4R4+3Une8hZa2rv83zEhN4qzibCrG53NL8RgmleRwftlwRg3POHnAe9WqvSoeQ0wkB9FXmNlFwCVB02fd/VBsY8lQdPaIHM4ekcOdC87meEsb63Ye4YWth9i4r5Y/vn2Q376655T5h6UmUxCc6pmdnkJmWjKZ6SlkpiYzLC2Z9NQkMlL+8nyoqpXcXUcozRtGcXb6Gb0fPjTs+DHe2FvDa7trWb+7hjf31dHc1n5ynpLcdMYVZDG9LI/cjBRyh6WSk5FCbkbqyZ9rQVYa+ZmhU2rP5J+XRCfSfxfmAPPCpp+IQRaRkzJSk0Nn6UwqOtlW09DMlv31bD9YT/WxZo4Eu1eqjzVzrKmVd2tbaGxpo6G5lcbmNo63tp88kHvCjypfAkLXGpTmD2N8YRbjCzMZX5TF2IJM8jLTGD4sleHDUskdlkJ6SmJu5TQ0t3KgromaxhZqGpqpbWyhpqGFrQfq2bivljffPUpjS+i6isy0ZKaXDufjl46nvCyPs0dkM6YgU1tw0meRnIV1LzATWBI0fcbM5rj7l2KaTKSDvMw0Zk0oYNaEzvdS6E57u9Pc1k5Dcxv//YcXGH32VPbVHmdfTehmPu9UH+OVd45Q39T1rUVz0lMozk2nJCeDEbnpFGenU5B94iK30MVthWFbQf15/UJLu/PGnlo27Ktl475a3qluoKr2OFV1xzsN2XFCdnoK543O5ZZZY5g6ejjTSnOZNCJHF9tJTESyBfJ+oNzd2wHM7CFgPaACIgkvKcnISEomIzWZMTlJzD+3pNM87s6h+tBd4moaW6gLHjUNLVQfa+bA0eMcqGti/a4aDh5tOvmffUdpKUkUZqUFWzEpoa2YjFRyh6WSlZ5Cdnpy8BwqNG3t7bS1h4pcQ3NraGyn4FFV28i2Aw20Pf0CECpkE0dkM7E4izlnFVIyPIMRORkUZKUyfFgaeZmhraaCTO1qkoET6S6sPEI3dILQbWVFzhhmdnJ4jEg0NrdRfazp5O6z0LhNoTGbquubqWlopq6xlZ2HGqg73kJtY8vJYTp6kpIUyjEiJ52xBVlMzmri/bOnM3V0LmPyM1UYJOFEUkD+L7DezFYSOgtrHvDFmKYSSWDD0pIpS8ukLD/yG2S1tzsNLW0ca2qlvqkV99B9r5PNSE420lOSOm09rFq1ivnTR8XiI4j0i0jOwvq1ma0idBwE4AvuXhXTVDFSXV3Ngw8+GO8YXaqpqWHnzp3xjtErgy2z8saW8sZWIuY97Q19zew5d3/X3ZcHjyoze64vKzWzAjN7xszeDp7zu5in3MzWmNlGM3vdzD4U1vegme0ws8rgUd6XPCIiEgV37/IBZBC6C+FrQH7wugAYD7zV3XKRPIBvAXcHr+8GvtnFPJOBScHr0cC7QF4w/SBwc2/XO2PGDE9UK1eujHeEXhtsmZU3tpQ3tuKZF1jnXfxN7WkX1ieBzwZ/vF8Na68DftDHunU9MD94/RCwCvhC+AzuviXs9T4zOwAUAzV9XLeIiPQDCxWXHmYw+7S7f79fV2pW4+55wWsDjpyY7mb+WYQKzVR3bzezB4HZQBPwHKGtmaZull1MMPRKSUnJjKVLl/bjJ+k/9fX1ZGdnxztGrwy2zMobW8obW/HMu2DBglfcvaJjeyQF5GNdtbv7L06z3LPAyC66vgw8FF4wzOyIu3c6DhL0jSK0hXKbu/8prK0KSCN0z/Zt7n5Pjx8EqKio8HXr1p1utrhYtWoV8+fPj3eMXhlsmZU3tpQ3tuKZ18y6LCCRnMY7M+x1BnAFoV1aPRYQd7+yhzD7zWyUu78bFIMD3cyXC/w38OUTxSN473eDl01m9nPg8xF8DhER6UeRnMb76fBpM8sD+rofaDlwG3Bv8PxYxxnMLA14FPiFu/+mQ9+J4mPADcCGPuYREZFeimbs5WPAxD6u915gmZl9AngHWARgZhXA37r7HUHbPKDQzG4Plrvd3SuBJWZWTOjCxkrgbyNZaVfXgUydOpWZM2fS0tLCkiVLOi1TXl5OeXk5DQ0NLFu2rFN/RUUF06ZNo7a2lkcffbRT/+zZsznnnHM4dOgQTzzReQzKefPmMXHiROrr67u8RuWKK65gzJgx7N69m+ee63z29MKFCxk5ciTbt2/n+eef79R/zTXXUFRUxObNm1mzZk2n/htvvJHhw4ezYcMGutq9t2jRIjIzM6msrKSysvKUvpqaGubOnUtqaipr165l48aNnZa//fbbAXjppZfYsmXLKX2pqanceuutAKxevZodO3ac0p+ZmcmiRYsAePbZZ9mz59TReHNzc7npppsAWLFiBVVVp16eVFhYyLXXXgvA448/zrZt2045j37kyJEsXLgQgEceeYS6urpTli8rK+PKK0Mb0suWLaOhoeGU/gkTJnDZZZcBsGTJElpaWk7pnzx5MnPmzAHo8nd7uu9eRkboRkyx/u5VVVWxYsWKTv29/e51vE4hlt89gFtvvbVP373S0lJgYL571dXVp/RH890L//nG+rt3ur97J0QymOLj/OUGUsnAuUD37xgBd68mtCusY/s64I7g9S+BX3az/OV9Wb+IiPRdJAfRLwubbCVURD7k7nfGMlgs6CB6/xpsmZU3tpQ3tgblQXR3X21mFwIfAT4I7AB+2/8RRURkMOm2gJjZZODDweMQ8DChLZYFA5RNREQSWE9bIG8BfwSucfetAGb2uQFJJSIiCa+nwRRvIjT+1Eoz+6mZXUHorCcREZHuC4i7/87dbwGmACsJjYs1wsz+w8yuHqB8IiKSoE47nLu7H3P3X7n7tUAZodvZfuE0i4mIyBnutAUknLsfcff73L3TNRwiIjK09KqAiIiInKACIiIiUVEBERGRqKiAiIhIVFRAREQkKiogIiISFRUQERGJigqIiIhERQVERESiogIiIiJRiUsBMbMCM3vGzN4OnvO7ma/NzCqDx/Kw9glm9mcz22pmD5tZ2sClFxERiN8WyN3Ac+4+CXgumO5Ko7uXB4/rwtq/CXzX3c8GjgCfiG1cERHpKF4F5HrgoeD1Q8ANkS5oZgZcDvwmmuVFRKR/mLsP/ErNatw9L3htwJET0x3mawUqgVbgXnf/nZkVAX8Ktj4wszHA7919WjfrWgwsBigpKZmxdOnS/v9A/aC+vp7s7Ox4x+iVwZZZeWNLeWMrnnkXLFjwirtXdOpw95g8gGeBDV08rgdqOsx7pJv3KA2eJwI7gbOAImBr2DxjgA2RZJoxY4YnqpUrV8Y7Qq8NtszKG1vKG1vxzAus8y7+pvZ0T/Q+cfcru+szs/1mNsrd3zWzUcCBbt5jb/C83cxWARcCvwXyzCzF3VsJ3eRqb79/ABER6VG8joEsB24LXt8GPNZxBjPLN7P04HURMBfYFFTDlcDNPS0vIiKxFa8Cci9wlZm9DVwZTGNmFWZ2fzDPucA6M3uNUMG41903BX1fAO4ys61AIfCzAU0vIiKx24XVE3evBjrdFtfd1wF3BK9fAqZ3s/x2YFYsM4qISM90JbqIiERFBURERKKiAiIiIlFRARERkaiogIiISFRUQEREJCoqICIiEhUVEBERiYoKiIiIREUFREREoqICIiIiUVEBERGRqKiAiIhIVFRAREQkKiogIiISFRUQERGJigqIiIhERQVERESiEpcCYmYFZvaMmb0dPOd3Mc8CM6sMexw3sxuCvgfNbEdYX/lAfwYRkaEuXlsgdwPPufsk4Llg+hTuvtLdy929HLgcaACeDpvlf5/od/fKAcgsIiJh4lVArgceCl4/BNxwmvlvBn7v7g2xDCUiIpGLVwEpcfd3g9dVQMlp5r8F+HWHtm+Y2etm9l0zS+/3hCIi0iNz99i8sdmzwMguur4MPOTueWHzHnH3TsdBgr5RwOvAaHdvCWurAtKA+4Bt7n5PN8svBhYDlJSUzFi6dGnUnymW6uvryc7OjneMXhlsmZU3tpQ3tuKZd8GCBa+4e0WnDncf8AewGRgVvB4FbO5h3s8A9/XQPx94IpL1zpgxwxPVypUr4x2h1wZbZuWNLeWNrXjmBdZ5F39T47ULazlwW/D6NuCxHub9MB12XwVbIJiZETp+sqH/I4qISE/iVUDuBa4ys7eBK4NpzKzCzO4/MZOZjQfGAKs7LL/EzN4A3gCKgH8eiNAiIvIXKfFYqbtXA1d00b4OuCNseidQ2sV8l8cyn4iInJ6uRBcRkaiogIiISFRUQEREJCoqICIiEhUVEBERiYoKiIiIREUFREREoqICIiIiUVEBERGRqKiAiIhIVFRAREQkKiogIiISFRUQERGJigqIiIhERQVERESiogIiIiJRUQEREZGoqICIiEhUVEBERCQqcSkgZvZBM9toZu1mVtHDfAvNbLOZbTWzu8PaJ5jZn4P2h80sbWCSi4jICfHaAtkA3AQ8390MZpYM/BB4H3Ae8GEzOy/o/ibwXXc/GzgCfCK2cUVEpKO4FBB3f9PdN59mtlnAVnff7u7NwFLgejMz4HLgN8F8DwE3xCysiIh0KSXeAXpQCuwOm94DXAwUAjXu3hrWXtrdm5jZYmBxMFlvZqcrXPFSBByKd4heGmyZlTe2lDe24pl3XFeNMSsgZvYsMLKLri+7+2OxWm9H7n4fcN9ArS9aZrbO3bs9HpSIBltm5Y0t5Y2tRMwbswLi7lf28S32AmPCpsuCtmogz8xSgq2QE+0iIjKAEvk03rXApOCMqzTgFmC5uzuwErg5mO82YMC2aEREJCRep/HeaGZ7gNnAf5vZU0H7aDN7EiDYuvgU8BTwJrDM3TcGb/EF4C4z20romMjPBvozxEDC72brwmDLrLyxpbyxlXB5LfQPvYiISO8k8i4sERFJYCogIiISFRWQODCzB8zsgJltCGsrMLNnzOzt4Dk/nhnDmdkYM1tpZpuCIWg+E7QnZGYzyzCzl83stSDv14P2hB4Cx8ySzWy9mT0RTCdsXjPbaWZvmFmlma0L2hLy+wBgZnlm9hsze8vM3jSz2Yma18zOCX6uJx51ZvbZRMyrAhIfDwILO7TdDTzn7pOA54LpRNEK/C93Pw+4BLgzGFYmUTM3AZe7+wVAObDQzC4h8YfA+QyhE0ZOSPS8C9y9POzahET9PgB8D1jh7lOACwj9nBMyr7tvDn6u5cAMoAF4lETM6+56xOEBjAc2hE1vBkYFr0cBm+OdsYfsjwFXDYbMQCbwKqFRDA4BKUH7bOCpeOcLy1lG6I/C5cATgCV43p1AUYe2hPw+AMOBHQQnDSV63g4ZrwZeTNS82gJJHCXu/m7wugooiWeY7pjZeOBC4M8kcOZgd1AlcAB4BthGL4bAiYN/A/4BaA+mezVkTxw48LSZvRIMFwSJ+32YABwEfh7sIrzfzLJI3LzhbgF+HbxOuLwqIAnIQ/9iJNz51WaWDfwW+Ky714X3JVpmd2/z0C6AMkIDc06Jb6Lumdk1wAF3fyXeWXrhUne/iNBo2Xea2bzwzgT7PqQAFwH/4e4XAsfosPsnwfICEBzzug74r459iZJXBSRx7DezUQDB84E45zmFmaUSKh5L3P2RoDmhMwO4ew2hkQtmEwyBE3Ql0hA4c4HrzGwnoVGnLye0zz5R8+Lue4PnA4T2z88icb8Pe4A97v7nYPo3hApKouY94X3Aq+6+P5hOuLwqIIljOaFhWSDBhmcJhtD/GfCmu38nrCshM5tZsZnlBa+HETpe8yYJOgSOu3/R3cvcfTyhXRZ/cPdbSdC8ZpZlZjknXhPaT7+BBP0+uHsVsNvMzgmargA2kaB5w3yYv+y+ggTMqyvR48DMfg3MJzQ8837gq8DvgGXAWOAdYJG7H45TxFOY2aXAH4E3+Ms++i8ROg6ScJnN7HxC94lJJvRP0jJ3v8fMJhL6D78AWA981N2b4pe0MzObD3ze3a9J1LxBrkeDyRTgV+7+DTMrJAG/DwBmVg7cD6QB24GPE3w3SMy8WcAuYKK71wZtCffzVQEREZGoaBeWiIhERQVERESiogIiIiJRUQEREZGoqICIiEhUVEBEBoCZ1cc7g0h/UwEREZGoqICIxImZXRvc72O9mT1rZiVBe3Fwv4eNwcB/75hZUbzzinSkAiISPy8AlwQD/C0lNBovhEYm+IO7TyU0btPYOOUT6VHK6WcRkRgpAx4OBsZLI3TPCoBLgRsB3H2FmR2JUz6RHmkLRCR+vg/8wN2nA58EMuKcR6RXVEBE4mc4fxmi/baw9heBRQBmdjUQ93tfi3RFgymKDAAzawf2hTV9h9BdEr9L6H7nfwBmuvt8MxtBaBjvEmANcA0wPhFG4hUJpwIikmDMLB1oc/dWM5tN6E565XGOJdKJDqKLJJ6xwDIzSwKagb+Jcx6RLmkLREREoqKD6CIiEhUVEBERiYoKiIiIREUFREREoqICIiIiUfn/YdCptIK1mUAAAAAASUVORK5CYII=",
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
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "from statsmodels.graphics.tsaplots import plot_acf,plot_pacf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
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
    "fig = plot_acf(df['Seasonal First Difference'].iloc[7:],lags=30,ax=ax1)\n",
    "ax2 = fig.add_subplot(212)\n",
    "fig = plot_pacf(df['Seasonal First Difference'].iloc[7:],lags=30,ax=ax2)"
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
       "  <th>Dep. Variable:</th>         <td>Cost</td>       <th>  No. Observations:  </th>   <td>77</td>   \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>            <td>ARIMA(2, 1, 1)</td>  <th>  Log Likelihood     </th> <td>-40.019</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>            <td>Sun, 20 Aug 2023</td> <th>  AIC                </th> <td>88.038</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                <td>22:39:55</td>     <th>  BIC                </th> <td>97.361</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Sample:</th>             <td>03-01-2017</td>    <th>  HQIC               </th> <td>91.764</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th></th>                   <td>- 07-01-2023</td>   <th>                     </th>    <td> </td>   \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>    <td> </td>   \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>ar.L1</th>  <td>    0.5481</td> <td>    0.466</td> <td>    1.177</td> <td> 0.239</td> <td>   -0.365</td> <td>    1.461</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>ar.L2</th>  <td>   -0.0608</td> <td>    0.388</td> <td>   -0.157</td> <td> 0.875</td> <td>   -0.820</td> <td>    0.699</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>ma.L1</th>  <td>    0.3949</td> <td>    0.471</td> <td>    0.839</td> <td> 0.401</td> <td>   -0.527</td> <td>    1.317</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>sigma2</th> <td>    0.1659</td> <td>    0.012</td> <td>   13.761</td> <td> 0.000</td> <td>    0.142</td> <td>    0.189</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "  <th>Ljung-Box (L1) (Q):</th>     <td>0.00</td>  <th>  Jarque-Bera (JB):  </th> <td>454.00</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(Q):</th>                <td>0.97</td>  <th>  Prob(JB):          </th>  <td>0.00</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Heteroskedasticity (H):</th> <td>66.79</td> <th>  Skew:              </th>  <td>-0.02</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(H) (two-sided):</th>    <td>0.00</td>  <th>  Kurtosis:          </th>  <td>14.97</td>\n",
       "</tr>\n",
       "</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step)."
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                               SARIMAX Results                                \n",
       "==============================================================================\n",
       "Dep. Variable:                   Cost   No. Observations:                   77\n",
       "Model:                 ARIMA(2, 1, 1)   Log Likelihood                 -40.019\n",
       "Date:                Sun, 20 Aug 2023   AIC                             88.038\n",
       "Time:                        22:39:55   BIC                             97.361\n",
       "Sample:                    03-01-2017   HQIC                            91.764\n",
       "                         - 07-01-2023                                         \n",
       "Covariance Type:                  opg                                         \n",
       "==============================================================================\n",
       "                 coef    std err          z      P>|z|      [0.025      0.975]\n",
       "------------------------------------------------------------------------------\n",
       "ar.L1          0.5481      0.466      1.177      0.239      -0.365       1.461\n",
       "ar.L2         -0.0608      0.388     -0.157      0.875      -0.820       0.699\n",
       "ma.L1          0.3949      0.471      0.839      0.401      -0.527       1.317\n",
       "sigma2         0.1659      0.012     13.761      0.000       0.142       0.189\n",
       "===================================================================================\n",
       "Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):               454.00\n",
       "Prob(Q):                              0.97   Prob(JB):                         0.00\n",
       "Heteroskedasticity (H):              66.79   Skew:                            -0.02\n",
       "Prob(H) (two-sided):                  0.00   Kurtosis:                        14.97\n",
       "===================================================================================\n",
       "\n",
       "Warnings:\n",
       "[1] Covariance matrix calculated using the outer product of gradients (complex-step).\n",
       "\"\"\""
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# For non-seasonal data\n",
    "#p=1, d=1, q=0 or 1\n",
    "from statsmodels.tsa.arima.model import ARIMA\n",
    "model=ARIMA(df['Cost'],order=(2,1,1))\n",
    "model_fit=model.fit()\n",
    "model_fit.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Date'>"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAArkAAAHgCAYAAAChN3UWAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAAsTAAALEwEAmpwYAABM/ElEQVR4nO3dd3ic1Zn+8ftMkUa9uxe5V1xlA4HQOywh2RR6IIWQ3RTSdvkl2YRs2m4KSUhCCEsKBJJACgkhBELv2MjGNi5g3LtVrJE0M9LU8/tjRkJg2WozGs3r7+e65tJoyvue8ety++g5zzHWWgEAAABO4sr2AAAAAIB0I+QCAADAcQi5AAAAcBxCLgAAAByHkAsAAADHIeQCAADAcTyZOGh1dbWtra3NxKEBAAAASdKqVauarLU1vT2XkZBbW1ur+vr6TBwaAAAAkCQZY3Ye6TnKFQAAAOA4hFwAAAA4DiEXAAAAjpORmlwAAAAcXTQa1Z49e9TZ2ZntoYx4Pp9PEyZMkNfr7fd7CLkAAABZsGfPHpWUlKi2tlbGmGwPZ8Sy1qq5uVl79uzRlClT+v0+yhUAAACyoLOzU1VVVQTcPhhjVFVVNeAZb0IuAABAlhBw+2cwv06EXAAAgGPUgQMHdOmll2ratGlaunSpLrjgAm3evHlAx/jWt76VodENDSEXAADgGGSt1bvf/W6ddtpp2rp1q1atWqVvf/vbOnjw4ICOQ8gFAADAiPHkk0/K6/Xq+uuv735s4cKFOvnkk/WFL3xB8+fP13HHHad7771XkrR//36dcsopWrRokebPn69nn31WN954ozo6OrRo0SJdccUV2foovaK7AgAAQJZ97W8btHFfW1qPOXdcqb76L/OO+Pz69eu1dOnSwx7/85//rDVr1mjt2rVqamrSsmXLdMopp+i3v/2tzj33XH3pS19SPB5XKBTSO9/5Tv3kJz/RmjVr0jr2dCDkAgAAoNtzzz2nyy67TG63W6NHj9app56ql19+WcuWLdOHPvQhRaNRXXLJJVq0aFG2h3pUhFwAAIAsO9qMa6bMmzdPf/zjH/v9+lNOOUXPPPOM/v73v+uaa67RZz/7WV199dUZHOHQUJMLAABwDDrjjDMUDod1++23dz+2bt06lZeX695771U8HldjY6OeeeYZLV++XDt37tTo0aP10Y9+VB/5yEe0evVqSZLX61U0Gs3WxzgiZnIBAACOQcYY3X///brhhhv0v//7v/L5fKqtrdUPf/hDBQIBLVy4UMYYfec739GYMWN055136rvf/a68Xq+Ki4t11113SZKuu+46LViwQEuWLNE999yT5U/1JmOtTftB6+rqbH19fdqPCwAA4BSbNm3SnDlzsj2MnNHbr5cxZpW1tq6311OuAAAAgJwTCMeO+jwhFwAAADnne4+8ftTnCbkAAADIOczkAgAAwHGChFwAAAA4DTO5AAAAcBxmcgEAANCrW265RXPmzNEVV1yR7aHoL3/5izZu3Njv1wfD8aM+T8gFAAA4Rt1666169NFH+7WJQyx29JnToRpoyKVcAQAAAIe5/vrrtW3bNp1//vn6/ve/r0suuUQLFizQCSecoHXr1kmSbrrpJl111VU66aSTdNVVV6mxsVH/+q//qmXLlmnZsmV6/vnnJUmBQEDXXnutjjvuOC1YsEB/+tOfJEkf//jHVVdXp3nz5umrX/1q97lvvPFGzZ07VwsWLNDnP/95vfDCC3rggQf0hS98QYsWLdLWrVv7HH8wcvSQy7a+AAAA2faPG6UDr6b3mGOOk87/nyM+fdttt+nhhx/Wk08+qa997WtavHix/vKXv+iJJ57Q1VdfrTVr1kiSNm7cqOeee04FBQW6/PLL9ZnPfEYnn3yydu3apXPPPVebNm3S17/+dZWVlenVV5OfoaWlRZL0zW9+U5WVlYrH4zrzzDO1bt06jR8/Xvfff79ee+01GWPk9/tVXl6uiy++WBdddJHe+9739uvj9VWTS8gFAAA4xj333HPds69nnHGGmpub1dbWJkm6+OKLVVBQIEl67LHH3lJS0NbWpkAgoMcee0y///3vux+vqKiQJN133326/fbbFYvFtH//fm3cuFFz586Vz+fThz/8YV100UW66KKLBjzecCyuaNwe9TWEXAAAgGw7yoxrthUVFXXfTyQSeumll+Tz+fp83/bt2/W9731PL7/8sioqKnTNNdeos7NTHo9HK1eu1OOPP64//vGP+slPfqInnnhiQGPqa9GZRE0uAADAMe+d73xn9+Kzp556StXV1SotLT3sdeecc45+/OMfd3/fVdJw9tln66c//Wn34y0tLWpra1NRUZHKysp08OBB/eMf/5CUrN9tbW3VBRdcoB/84Adau3atJKmkpETt7e39Gm9fpQoS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",
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
    "df['forecast']=model_fit.predict(start=30,end=78,dynamic=True)\n",
    "df[['Cost','forecast']].plot(figsize=(12,8))"
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
      "C:\\Users\\Vishi Ved\\AppData\\Roaming\\Python\\Python310\\site-packages\\statsmodels\\tsa\\base\\tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.\n",
      "  self._init_dates(dates, freq)\n",
      "C:\\Users\\Vishi Ved\\AppData\\Roaming\\Python\\Python310\\site-packages\\statsmodels\\tsa\\base\\tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.\n",
      "  self._init_dates(dates, freq)\n",
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
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAsEAAAHgCAYAAABEqbB6AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAAsTAAALEwEAmpwYAABTV0lEQVR4nO3dd3xcV5n/8e+Z0Uij3i0XudtxL0nsVMfppBJC2zRKIIW2u8Au7IYf7EJg6Sx9d0M2BLKQkEA6KYSE9ObYTtxr3Itkyaozkqaf3x93ZMtFssqVRpr5vF+v+5rR6M65Z3Qd59Hxc57HWGsFAAAAZBJPqicAAAAADDWCYAAAAGQcgmAAAABkHIJgAAAAZByCYAAAAGQcgmAAAABknKxUXLSiosJOmjQpFZcGAABABlm5cuVBa23l0a+nJAieNGmSVqxYkYpLAwAAIIMYY3Yd73XSIQAAAJBxCIIBAACQcQiCAQAAkHFSkhMMAACAnkWjUe3du1ehUCjVUxkR/H6/qqur5fP5enU+QTAAAMAwtHfvXhUWFmrSpEkyxqR6OsOatVYNDQ3au3evJk+e3Kv3kA4BAAAwDIVCIZWXlxMA94IxRuXl5X1aNScIBgAAGKYIgHuvrz8rgmAAAAB0q7a2Vtdee62mTp2qU089VZdffrm2bNnSpzG+853vDNLs+o8gGAAAAMdlrdX73/9+nXfeedq2bZtWrlyp7373uzpw4ECfxiEIBgAAwIjxwgsvyOfz6dOf/vSh1xYsWKAlS5boy1/+subOnat58+bpgQcekCTV1NRo6dKlWrhwoebOnatXXnlFt912mzo6OrRw4ULdcMMNqfoox6A6BAAAwDB3+5/Xa8P+VlfHnD22SF9/75wez1m3bp1OPfXUY15/+OGHtWrVKq1evVoHDx7U4sWLtXTpUt1333265JJL9NWvflXxeFzt7e0655xz9Mtf/lKrVq1ydf4DRRAMAACAPnn11Vd13XXXyev1qqqqSueee66WL1+uxYsX65Of/KSi0aiuvvpqLVy4MNVT7RZBMAAAwDB3ohXbwTJnzhw9+OCDvT5/6dKlevnll/Xkk0/qxhtv1D/90z/pYx/72CDOsP/ICQYAAMBxXXDBBQqHw7rzzjsPvbZmzRqVlJTogQceUDweV319vV5++WWddtpp2rVrl6qqqnTLLbfo5ptv1ttvvy1J8vl8ikajqfoYx8VKMAAAAI7LGKNHHnlEX/jCF/T9739ffr9fkyZN0k9/+lMFg0EtWLBAxhj94Ac/0OjRo3XPPffohz/8oXw+nwoKCvR///d/kqRbb71V8+fP1ymnnKJ77703xZ/KYay1Q37RRYsW2RUrVgz5dQEAAEaKjRs3atasWamexohyvJ+ZMWaltXbR0eeyEgwAAIC0E40n1BaOdft9coIBAACQdjbXBrTwm892+32CYAAAAKSdQKj7VWCJIBgAAABpKNhDKoREEAwAAIA0FAj1XJKNIBgAAABph5VgAAAA9MvPf/5zzZo1SzfccEOqp6JHH31UGzZs6PX55AQDAACgX/77v/9bzz77bK8aXMRiPQedA9WfIDjb232oSxAMAACAY3z605/W9u3bddlll+k///M/dfXVV2v+/Pk644wztGbNGknSN77xDX30ox/V2WefrY9+9KOqr6/XBz/4QS1evFiLFy/Wa6+9JkkKBoP6xCc+oXnz5mn+/Pl66KGHJEmf+cxntGjRIs2ZM0df//rXD137tttu0+zZszV//nx96Utf0uuvv67HH39cX/7yl7Vw4UJt27bthPMPhKIq9HffEoNmGQAAAMPd07dJtWvdHXP0POmy73X77TvuuEN/+ctf9MILL+j222/XySefrEcffVTPP/+8Pvaxj2nVqlWSpA0bNujVV19Vbm6urr/+en3xi1/UkiVLtHv3bl1yySXauHGjvvWtb6m4uFhr1zqfoampSZL07W9/W2VlZYrH47rwwgu1Zs0ajRs3To888og2bdokY4yam5tVUlKiq666SldeeaU+9KEP9erjBcMxFRAEAwAAoL9effXVQ6u3F1xwgRoaGtTa2ipJuuqqq5SbmytJeu65545IWWhtbVUwGNRzzz2n+++//9DrpaWlkqQ//vGPuvPOOxWLxVRTU6MNGzZo9uzZ8vv9uummm3TllVfqyiuv7NecA6EYK8EAAAAjWg8rtqmWn59/6HkikdCbb74pv99/wvft2LFDP/rRj7R8+XKVlpbqxhtvVCgUUlZWlt566y397W9/04MPPqhf/vKXev755/s8r2AopoKc7kNdcoIBAADQo3POOefQ5rgXX3xRFRUVKioqOua897znPfrFL35x6OvOlImLL75Y//Vf/3Xo9aamJrW2tio/P1/FxcU6cOCAnn76aUlO/nBL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",
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
    "model=sm.tsa.statespace.SARIMAX(df['Cost'],order=(2,1,1), seasonal_order=(2,1,1,6))\n",
    "results=model.fit()\n",
    "df['forecast']=results.predict(start=30,end=78,dynamic=True)\n",
    "df[['Cost','forecast']].plot(figsize=(12,8))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pandas.tseries.offsets import DateOffset\n",
    "future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
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
     "execution_count": 37,
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
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "future_df=pd.concat([df,future_datest_df])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
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
     "execution_count": 39,
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
    "future_df['forecast'] = results.predict(start = 79, end = 400, dynamic= True)  \n",
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
