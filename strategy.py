#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Camarilla pivot S3/R3 breakout with volume confirmation and 1-day EMA34 trend filter
# - Long when price breaks above Camarilla R3 with volume expansion and price above 1-day EMA34
# - Short when price breaks below Camarilla S3 with volume expansion and price below 1-day EMA34
# - Exit when price crosses back below/above 1-day EMA34
# - Volume filter requires current volume > 1.3x 20-period average
# - Designed to capture strong trends while avoiding whipsaws in ranging markets
# - Target: 75-200 total trades over 4 years (19-50/year) with 0.25 position sizing

name = "4h_Camarilla_R3S3_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and EMA calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (S3, R3) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and ranges
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels (S3 and R3)
    s3 = close_1d - (range_hl * 1.1 / 6)
    r3 = close_1d + (range_hl * 1.1 / 6)
    
    # Calculate 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe (wait for completed daily bar)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    ema_34_1d_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filters (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)  # Volume confirmation
    volume_expansion = volume > np.roll(volume, 1)  # Current volume > previous
    volume_expansion[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(s3_4h[i]) or np.isnan(r3_4h[i]) or 
            np.isnan(ema_34_1d_4h[i]) or np.isnan(volume_filter[i]) or np.isnan(volume_expansion[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above Camarilla R3 with volume expansion and above EMA34
            if close[i] > r3_4h[i] and volume_expansion[i] and close[i] > ema_34_1d_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below Camarilla S3 with volume expansion and below EMA34
            elif close[i] < s3_4h[i] and volume_expansion[i] and close[i] < ema_34_1d_4h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA34
            if close[i] < ema_34_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above EMA34
            if close[i] > ema_34_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals