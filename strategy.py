#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above 12h Camarilla R3 AND price > 1d EMA34 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below 12h Camarilla S3 AND price < 1d EMA34 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses 12h Camarilla midpoint (R3/S3 midpoint) OR volume < avg_volume(20)
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 50-150 total trades over 4 years (12-37/year)
# Camarilla levels from 12h provide intraday support/resistance; 1d EMA34 filters primary trend; volume spike confirms breakout strength
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike"
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
    
    # Calculate 12h Camarilla levels (R3, S3, midpoint)
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    rango = high_series - low_series
    camarilla_r3 = close_series + 1.1 * rango / 2
    camarilla_s3 = close_series - 1.1 * rango / 2
    camarilla_mid = (camarilla_r3 + camarilla_s3) / 2
    camarilla_r3_vals = camarilla_r3.values
    camarilla_s3_vals = camarilla_s3.values
    camarilla_mid_vals = camarilla_mid.values
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_vals[i]) or np.isnan(camarilla_s3_vals[i]) or 
            np.isnan(camarilla_mid_vals[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R3, above 1d EMA34, volume confirmation
            if close[i] > camarilla_r3_vals[i] and close[i] > ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3, below 1d EMA34, volume confirmation
            elif close[i] < camarilla_s3_vals[i] and close[i] < ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Camarilla midpoint OR volume drops below average
            if close[i] < camarilla_mid_vals[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above Camarilla midpoint OR volume drops below average
            if close[i] > camarilla_mid_vals[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals