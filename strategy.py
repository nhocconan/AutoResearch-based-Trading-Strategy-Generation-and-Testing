#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot levels from daily with volume confirmation and chop filter
# Long when price touches Camarilla S1 (support) with volume >1.5x 24-period average and chop >61.8 (range)
# Short when price touches Camarilla R1 (resistance) with volume >1.5x 24-period average and chop >61.8 (range)
# Exit when price crosses Camarilla P (pivot point)
# Uses daily Camarilla levels as structure, chop filter to avoid trending markets, volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: range = high - low
    # Pivot = (high + low + close) / 3
    # S1 = close - (range * 1.1 / 12)
    # R1 = close + (range * 1.1 / 12)
    range_1d = high_1d - low_1d
    camarilla_p = (high_1d + low_1d + close_1d) / 3
    camarilla_s1 = close_1d - (range_1d * 1.1 / 12)
    camarilla_r1 = close_1d + (range_1d * 1.1 / 12)
    
    # Calculate 24-period volume average for 12h timeframe (24 * 12h = 12 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate 12-period chop index (choppiness) for regime filter
    # Chop = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    # Using simplified version: chop > 61.8 = range, chop < 38.2 = trend
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[0], tr2])  # align with index
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(pd.Series(atr).rolling(window=14, min_periods=14).sum().values / 
                          np.maximum(max_high - min_low, 1e-10)) / np.log10(14)
    
    # Align daily Camarilla levels to 12h timeframe
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for 24-period volume and 14-period chop
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_p_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(vol_ma_24[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long setup: touch S1 with volume confirmation in ranging market
            if (abs(price - camarilla_s1_aligned[i]) < 0.001 * camarilla_s1_aligned[i] and  # within 0.1%
                vol_current > 1.5 * vol_ma_24[i] and
                chop[i] > 61.8):  # ranging market
                position = 1
                signals[i] = position_size
            # Short setup: touch R1 with volume confirmation in ranging market
            elif (abs(price - camarilla_r1_aligned[i]) < 0.001 * camarilla_r1_aligned[i] and  # within 0.1%
                  vol_current > 1.5 * vol_ma_24[i] and
                  chop[i] > 61.8):  # ranging market
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses pivot point
            if price > camarilla_p_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses pivot point
            if price < camarilla_p_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_Daily_Volume_Chop"
timeframe = "12h"
leverage = 1.0