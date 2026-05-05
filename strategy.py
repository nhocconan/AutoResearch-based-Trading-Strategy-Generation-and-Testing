#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above 4h Camarilla R3 AND price > 1d EMA34 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below 4h Camarilla S3 AND price < 1d EMA34 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses 4h Camarilla midpoint (R3/S3 midpoint) OR volume < avg_volume(20)
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 75-200 total trades over 4 years (19-50/year)
# Camarilla levels from 4h provide intraday support/resistance; 1d EMA34 filters primary trend; volume spike confirms breakout strength
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Camarilla levels (using typical price for pivot)
    typical_price = (high + low + close) / 3.0
    tp_series = pd.Series(typical_price)
    pivot = tp_series.rolling(window=20, min_periods=20).mean().values
    range_hl = high - low
    range_series = pd.Series(range_hl)
    avg_range = range_series.rolling(window=20, min_periods=20).mean().values
    
    # Camarilla levels
    R3 = pivot + (avg_range * 1.1 / 4.0)
    S3 = pivot - (avg_range * 1.1 / 4.0)
    midpoint = (R3 + S3) / 2.0  # (R3+S3)/2 = pivot
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA34
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
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or np.isnan(midpoint[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3, above 1d EMA34, volume confirmation
            if close[i] > R3[i] and close[i] > ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3, below 1d EMA34, volume confirmation
            elif close[i] < S3[i] and close[i] < ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below midpoint OR volume drops below average
            if close[i] < midpoint[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above midpoint OR volume drops below average
            if close[i] > midpoint[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals