#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour TRIX momentum with daily volume spike and choppiness regime filter
# Long when TRIX crosses above zero, volume spike, and market is trending (CHOP < 38.2)
# Short when TRIX crosses below zero, volume spike, and market is trending (CHOP < 38.2)
# TRIX filters noise and captures sustained momentum
# Volume spike confirms institutional participation in the move
# Choppiness regime filter avoids whipsaws in ranging markets
# Targets 80-160 total trades over 4 years (20-40/year) for optimal balance

name = "4h_TRIX_ZeroCross_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for choppiness filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate TRIX(12) - triple exponential moving average rate of change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3.pct_change())
    trix_values = trix.values
    
    # Calculate choppiness index from daily data
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high) - min(low))))
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).sum()
    
    highest_high = df_1d['high'].rolling(window=14, min_periods=14).max()
    lowest_low = df_1d['low'].rolling(window=14, min_periods=14).min
    
    chop = 100 * np.log10(atr14 / (np.log10(14) * (highest_high - lowest_low)))
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trix_values[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trix_val = trix_values[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        
        # TRIX zero cross signals with volume and chop filter
        if i > 0 and not np.isnan(trix_values[i-1]):
            trix_prev = trix_values[i-1]
            # Enter long: TRIX crosses above zero, volume spike, trending market (CHOP < 38.2)
            if trix_prev <= 0 and trix_val > 0 and vol_spike and chop_val < 38.2:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero, volume spike, trending market (CHOP < 38.2)
            elif trix_prev >= 0 and trix_val < 0 and vol_spike and chop_val < 38.2:
                signals[i] = -0.25
                position = -1
        
        # Exit conditions: TRIX returns to zero or chop increases (ranging market)
        if position == 1:
            if trix_val <= 0 or chop_val >= 61.8:  # exit when momentum fades or market ranges
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if trix_val >= 0 or chop_val >= 61.8:  # exit when momentum fades or market ranges
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals