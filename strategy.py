#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and 1d choppiness regime filter
# Long when price breaks above 12h Camarilla R3 level AND 1d volume > 2x 20-period average AND 1d chop < 50 (trending)
# Short when price breaks below 12h Camarilla S3 level AND 1d volume > 2x 20-period average AND 1d chop < 50 (trending)
# Exit when price crosses 12h Camarilla pivot point (mean reversion)
# Uses 12h primary timeframe with 1d HTF for volume and chop filters
# Camarilla levels provide clear breakout zones based on previous day's range
# Volume confirmation filters low-momentum breakouts
# Chop filter ensures we only trade in trending markets, reducing whipsaw in ranges
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_Camarilla_R3S3_Breakout_1dVol_Chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume and chop filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume MA(20) for volume spike filter
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol_1d > (2.0 * vol_ma_20)
    
    # Calculate 1d choppiness index (CHOP) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # ATR(14) - sum of TR over 14 periods
    atr_14 = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 14:
            atr_14[i] = np.nan
        else:
            atr_14[i] = np.nansum(tr[i-13:i+1])
    
    # Chop = 100 * log10(sum(TR(14)) / (max_high - min_low)) / log10(14)
    max_high = np.zeros_like(close_1d)
    min_low = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 14:
            max_high[i] = np.nan
            min_low[i] = np.nan
        else:
            max_high[i] = np.nanmax(high_1d[i-13:i+1])
            min_low[i] = np.nanmin(low_1d[i-13:i+1])
    
    chop = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if np.isnan(atr_14[i]) or np.isnan(max_high[i]) or np.isnan(min_low[i]) or max_high[i] == min_low[i]:
            chop[i] = np.nan
        else:
            chop[i] = 100 * np.log10(atr_14[i] / (max_high[i] - min_low[i])) / np.log10(14)
    
    # Align 1d indicators to 12h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get 1d data ONCE before loop for Camarilla levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Camarilla R3 and S3 levels: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3 = close_1d + (1.1 * (high_1d - low_1d) / 2)
    camarilla_s3 = close_1d - (1.1 * (high_1d - low_1d) / 2)
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3  # Standard pivot point
    
    # Align to 12h timeframe (using previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND volume spike AND chop < 50 (trending)
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike_aligned[i] and 
                chop_aligned[i] < 50):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND volume spike AND chop < 50 (trending)
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike_aligned[i] and 
                  chop_aligned[i] < 50):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot (mean reversion)
            if close[i] < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot (mean reversion)
            if close[i] > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals