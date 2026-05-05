#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume spike confirmation
# Long when price breaks above 1d Camarilla R3 AND price > 1w EMA34 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below 1d Camarilla S3 AND price < 1w EMA34 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses 1d Camarilla H3/L3 midpoint OR volume < avg_volume(20)
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 50-150 total trades over 4 years (12-37/year)
# Camarilla levels from 1d provide robust intraday support/resistance; 1w EMA34 filters primary trend; volume spike confirms breakout strength
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)

name = "12h_Camarilla_R3S3_Breakout_1wEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 1 day for Camarilla calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    #          H3 = (high+low)/2 + 1.1*(high-low)/2, L3 = (high+low)/2 - 1.1*(high-low)/2
    # We use previous day's values to avoid look-ahead
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # Shift by 1 to use previous day's values
    high_1d_prev = high_1d_series.shift(1).values
    low_1d_prev = low_1d_series.shift(1).values
    close_1d_prev = close_1d_series.shift(1).values
    
    # Calculate Camarilla levels for each bar (using previous day's OHLC)
    camarilla_R3 = close_1d_prev + 1.1 * (high_1d_prev - low_1d_prev)
    camarilla_S3 = close_1d_prev - 1.1 * (high_1d_prev - low_1d_prev)
    camarilla_H3 = (high_1d_prev + low_1d_prev) / 2 + 1.1 * (high_1d_prev - low_1d_prev) / 2
    camarilla_L3 = (high_1d_prev + low_1d_prev) / 2 - 1.1 * (high_1d_prev - low_1d_prev) / 2
    camarilla_mid = (camarilla_H3 + camarilla_L3) / 2  # Midpoint for exit
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough for EMA34
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_R3[i]) or np.isnan(camarilla_S3[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R3, above 1w EMA34, volume confirmation
            if close[i] > camarilla_R3[i] and close[i] > ema34_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3, below 1w EMA34, volume confirmation
            elif close[i] < camarilla_S3[i] and close[i] < ema34_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Camarilla midpoint OR volume drops below average
            if close[i] < camarilla_mid[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above Camarilla midpoint OR volume drops below average
            if close[i] > camarilla_mid[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals