#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Camarilla pivot levels (R3/S3) with volume spike and chop regime filter
# - Uses 1d Camarilla R3/S3 levels for breakout signals
# - Uses 12h volume spike (2x 20-period average) for confirmation
# - Uses 12h Choppiness Index (14-period) < 38.2 to filter for trending markets only
# - Enters long when price breaks above 1d Camarilla R3 with volume and trending regime
# - Enters short when price breaks below 1d Camarilla S3 with volume and trending regime
# - Exits when price returns to 1d Camarilla pivot (midpoint) or opposite level
# - Designed to capture strong trending moves with institutional level respect
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_1dCamarilla_R3S3_Volume_Chop"
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
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R3, S3, pivot)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for Camarilla calculation
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.concatenate([[np.nan], close_1d[:-1]])),
                                  np.absolute(np.concatenate([[np.nan], close_1d[:-1]]) - low_1d)))
    # For first element, use high-low
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Calculate pivot point (PP)
    pp_1d = (high_1d + low_1d + close_1d) / 3
    
    # Calculate R3 and S3 levels
    r3_1d = pp_1d + 1.1 * (high_1d - low_1d)
    s3_1d = pp_1d - 1.1 * (high_1d - low_1d)
    
    # Align 1d Camarilla levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3_1d)
    pp_12h = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # Volume filter (12h timeframe) - spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Volume spike confirmation
    
    # Choppiness Index filter (12h timeframe) - trending regime
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr = np.zeros_like(close)
        
        for i in range(len(close)):
            if i == 0:
                tr[i] = high[i] - low[i]
            else:
                tr[i] = max(high[i] - low[i], 
                           abs(high[i] - close[i-1]), 
                           abs(low[i] - close[i-1]))
        
        # Calculate ATR using Wilder's smoothing
        atr[period-1] = np.mean(tr[1:period+1])
        for i in range(period, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Calculate Choppiness Index
        chop = np.zeros_like(close)
        for i in range(period, len(close)):
            atr_sum = np.sum(atr[i-period+1:i+1])
            hh = np.max(high[i-period+1:i+1])
            ll = np.min(low[i-period+1:i+1])
            if hh != ll and atr_sum > 0:
                chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
            else:
                chop[i] = 50  # Neutral value when undefined
        return chop
    
    chop_values = calculate_chop(high, low, close, 14)
    chop_filter = chop_values < 38.2  # Trending regime (CHOP < 38.2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(pp_12h[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above 1d Camarilla R3 with volume and trending regime
            if close[i] > r3_12h[i] and volume_spike[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below 1d Camarilla S3 with volume and trending regime
            elif close[i] < s3_12h[i] and volume_spike[i] and chop_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot OR breaks below S3
            if close[i] < pp_12h[i] or close[i] < s3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to pivot OR breaks above R3
            if close[i] > pp_12h[i] or close[i] > r3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals