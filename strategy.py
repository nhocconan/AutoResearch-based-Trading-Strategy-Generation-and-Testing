#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above 1d Camarilla R3 AND price > 1d EMA34 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 1d Camarilla S3 AND price < 1d EMA34 AND volume > 1.5 * avg_volume(20)
# Exit when price crosses 1d Camarilla H3/L3 midpoint OR volume < avg_volume(20)
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 50-150 total trades over 4 years (12-37/year)
# Camarilla levels from 1d provide intraday support/resistance; 1d EMA34 filters primary trend; volume spike confirms breakout strength
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
    
    # Calculate Camarilla levels from daily OHLC
    # Need to get daily OHLC from 12h data - we'll use rolling window of 2 bars (since 2*12h = 1d)
    # But better to get actual 1d data for accuracy
    
    # Get 1d data ONCE before loop for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA34
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    R3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    S3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    H3 = close_1d + (high_1d - low_1d) * 1.1 / 6
    L3 = close_1d - (high_1d - low_1d) * 1.1 / 6
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume (using 12h data)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R3, above 1d EMA34, volume confirmation
            if close[i] > R3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3, below 1d EMA34, volume confirmation
            elif close[i] < S3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below (H3+L3)/2 midpoint OR volume drops below average
            midpoint = (H3_aligned[i] + L3_aligned[i]) / 2
            if close[i] < midpoint or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above (H3+L3)/2 midpoint OR volume drops below average
            midpoint = (H3_aligned[i] + L3_aligned[i]) / 2
            if close[i] > midpoint or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals