#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour TRIX + Volume Spike + Choppiness Regime
# Long when TRIX > 0 (bullish momentum) + volume spike + choppiness > 61.8 (range)
# Short when TRIX < 0 (bearish momentum) + volume spike + choppiness > 61.8 (range)
# TRIX filters noise and identifies sustained momentum, effective in choppy markets
# Volume spike confirms institutional participation
# Choppiness regime filter ensures we only trade in ranging conditions where mean reversion works
# Targets 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "12h_TRIX_Volume_Chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate TRIX (15,9,9) on 12h close
    # TRIX = EMA(EMA(EMA(close, 15), 9), 9) - 1 period ago
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix[0] = 0  # first value has no previous
    
    # Calculate Choppiness Index (14) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]  # first TR
    
    # Sum of TR over 14 periods
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(atr14 / (hh14 - ll14)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((hh14 - ll14) == 0, 100, chop)
    chop = np.where(np.isnan(chop), 50, chop)
    
    # Align choppiness to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trix[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trix_val = trix[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: TRIX > 0 + volume spike + chop > 61.8 (range)
            if trix_val > 0 and vol_spike and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX < 0 + volume spike + chop > 61.8 (range)
            elif trix_val < 0 and vol_spike and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX <= 0 OR chop <= 61.8 (trending)
            if trix_val <= 0 or chop_val <= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX >= 0 OR chop <= 61.8 (trending)
            if trix_val >= 0 or chop_val <= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals