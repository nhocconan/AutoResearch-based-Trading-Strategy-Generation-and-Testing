#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Choppiness_Reversal_v1"
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
    
    # Get 1d data for Choppiness Index and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Choppiness Index on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range (TR)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(tr14) / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    choppiness = np.where(
        (range_hl > 0) & (~np.isnan(tr_sum)),
        100 * np.log10(tr_sum / range_hl) / np.log10(14),
        50  # neutral when range is zero or invalid
    )
    
    # Align Choppiness Index to 4h timeframe
    choppiness_aligned = align_htf_to_ltf(prices, df_1d, choppiness)
    
    # Calculate 4h EMA20 for trend filter
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume spike filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need enough data for EMA20 and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(choppiness_aligned[i]) or 
            np.isnan(ema20[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop = choppiness_aligned[i]
        ema20_val = ema20[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Chop > 61.8 (ranging) + price near low + volume spike
            if chop > 61.8 and close[i] <= low[i] * 1.005 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Chop > 61.8 (ranging) + price near high + volume spike
            elif chop > 61.8 and close[i] >= high[i] * 0.995 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Chop < 38.2 (trending) or price reaches high
            if chop < 38.2 or close[i] >= high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Chop < 38.2 (trending) or price reaches low
            if chop < 38.2 or close[i] <= low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals