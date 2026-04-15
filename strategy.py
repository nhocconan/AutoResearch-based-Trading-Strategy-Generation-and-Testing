#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX + volume spike + 1d Choppiness regime filter
# Uses TRIX (triple-smoothed EMA) for momentum with reduced whipsaw, volume to confirm momentum strength,
# and 1d Choppiness Index to avoid ranging markets. Works in both bull and bear by
# only taking signals when TRIX crosses zero in the direction of the 1d EMA50 trend.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for TRIX and price action
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Load 1d data for Choppiness Index and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate TRIX (15-period triple-smoothed EMA) on 4h
    ema1 = pd.Series(close_4h).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # First value
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Choppiness Index (14-period) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (hh_14 - ll_14 + 1e-10)) / np.log10(14)
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_4h, trix)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: TRIX crosses above zero + volume spike + chop < 61.8 (trending) + price above EMA50_1d
        if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            chop_aligned[i] < 61.8 and
            close[i] > ema50_1d_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: TRIX crosses below zero + volume spike + chop < 61.8 (trending) + price below EMA50_1d
        elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              chop_aligned[i] < 61.8 and
              close[i] < ema50_1d_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse TRIX signal or chop > 61.8 (ranging market)
        elif position == 1 and (trix_aligned[i] < 0 or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (trix_aligned[i] > 0 or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_TRIX_Volume_Chop_Filter"
timeframe = "4h"
leverage = 1.0