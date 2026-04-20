#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter with 1d CAMARILLA pivot breakouts
# Choppiness Index (CHOP) > 61.8 indicates ranging market (mean reversion opportunity)
# CAMARILLA pivot levels (R1, S1) from daily timeframe provide entry/exit levels
# Volume > 1.5x 20-period average confirms institutional participation
# Designed for 12h timeframe with selective entries to avoid overtrading
# Target: 12-37 trades per year per symbol (48-148 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for CAMARILLA pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate CAMARILLA pivot levels for 1d
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align CAMARILLA levels to 12h timeframe (wait for daily close)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate Choppiness Index on 12h timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) for CHOP denominator
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sumTR14 / (ATR14 * 14)) / log10(14)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    chop_raw = 100 * np.log10(sum_tr14 / (atr14 * 14 + epsilon)) / np.log10(14)
    # Handle invalid values (when ATR is 0 or sum_TR is 0)
    chop = np.where((atr14 > epsilon) & (sum_tr14 > 0), chop_raw, 50.0)  # default to neutral 50
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or \
           np.isnan(s1_1d_aligned[i]) or np.isnan(chop[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Choppiness regime: > 61.8 = ranging (mean reversion opportunity)
        is_ranging = chop[i] > 61.8
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Long entry: price crosses above S1 in ranging market + volume
            long_signal = is_ranging and has_volume and (price > s1_1d_aligned[i]) and (close[i-1] <= s1_1d_aligned[i-1])
            
            # Short entry: price crosses below R1 in ranging market + volume
            short_signal = is_ranging and has_volume and (price < r1_1d_aligned[i]) and (close[i-1] >= r1_1d_aligned[i-1])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below pivot or chop drops below 38.2 (trending)
            exit_signal = (price < pivot_1d_aligned[i]) or (chop[i] < 38.2)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above pivot or chop drops below 38.2 (trending)
            exit_signal = (price > pivot_1d_aligned[i]) or (chop[i] < 38.2)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_CHOP_Ranging_CAMARILLA_S1R1_Breakout"
timeframe = "12h"
leverage = 1.0