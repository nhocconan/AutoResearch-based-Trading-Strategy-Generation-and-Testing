#!/usr/bin/env python3
# 6h_1W_1D_LinearRegression_Channel_Breakout_VolumeFilter
# Hypothesis: 6-hour breakouts from weekly linear regression channel with volume confirmation.
# Uses weekly linear regression of closing prices to define dynamic support/resistance channels.
# In bull markets: price breaks above upper channel line + volume = continuation long.
# In bear markets: price breaks below lower channel line + volume = continuation short.
# Includes volume filter to reduce false breakouts. Targets 12-30 trades per year.

name = "6h_1W_1D_LinearRegression_Channel_Breakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from scipy import stats
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Weekly data for linear regression channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate linear regression for last 30 weekly closes
    # Using weekly close prices for the regression
    weekly_closes = df_1w['close'].values
    
    # Initialize arrays for channel bounds
    upper_channel = np.full(len(weekly_closes), np.nan)
    lower_channel = np.full(len(weekly_closes), np.nan)
    
    # Calculate linear regression for each window of 30 weeks
    for i in range(30, len(weekly_closes)):
        y = weekly_closes[i-30:i]
        x = np.arange(30)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        # Predict next value (forward projection)
        pred = slope * 30 + intercept
        # Channel width based on standard error
        channel_width = std_err * 1.5
        upper_channel[i] = pred + channel_width
        lower_channel[i] = pred - channel_width
    
    # Align weekly channel to 6h timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_1w, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1w, lower_channel)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(upper_channel_aligned[i]) or 
            np.isnan(lower_channel_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper channel + volume spike
            if close[i] > upper_channel_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower channel + volume spike
            elif close[i] < lower_channel_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters channel or breaks below lower channel
            if close[i] < upper_channel_aligned[i] and close[i] > lower_channel_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters channel or breaks above upper channel
            if close[i] < upper_channel_aligned[i] and close[i] > lower_channel_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals