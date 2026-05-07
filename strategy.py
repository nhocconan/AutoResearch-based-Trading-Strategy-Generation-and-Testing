#!/usr/bin/env python3
name = "4h_Trix_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate TRIX (15,9) on 4h close
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.pct_change(periods=1)
    trix_signal = trix.ewm(span=9, adjust=False, min_periods=9).mean()
    trix_hist = (trix - trix_signal).values
    
    # Daily chop regime: CHOP(14) > 61.8 = range, < 38.2 = trend
    atr_1d = pd.Series(np.maximum(np.maximum(df_1d['high'] - df_1d['low'], 
                                              np.abs(df_1d['high'] - df_1d['close'].shift(1))),
                                  np.abs(df_1d['low'] - df_1d['close'].shift(1)))).rolling(14, min_periods=14).mean()
    max_high_1d = df_1d['high'].rolling(14, min_periods=14).max()
    min_low_1d = df_1d['low'].rolling(14, min_periods=14).min()
    chop = 100 * np.log10((atr_1d.sum() / (max_high_1d - min_low_1d))) / np.log10(14)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Volume spike detection on 4h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trix_hist[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trix_long = trix_hist[i] > 0 and trix_hist[i-1] <= 0
        trix_short = trix_hist[i] < 0 and trix_hist[i-1] >= 0
        vol_spike = vol_ratio[i] > 2.0
        chop_condition = chop_aligned[i] > 61.8  # range regime for mean reversion
        
        if position == 0:
            if trix_long and vol_spike and chop_condition:
                signals[i] = 0.25
                position = 1
            elif trix_short and vol_spike and chop_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            if trix_hist[i] < 0:  # TRIX histogram crosses below zero
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if trix_hist[i] > 0:  # TRIX histogram crosses above zero
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX histogram crossovers with volume spikes in choppy (range) markets
# - TRIX (15,9) histogram zero cross provides momentum signal
# - Volume spike (>2x 20-period average) confirms conviction
# - Chop regime filter (CHOP(14) > 61.8) ensures ranging market for mean reversion
# - Works in both bull and bear markets as it captures short-term reversals in ranges
# - Exit on TRIX histogram sign change
# - Position size 0.25 limits risk and reduces trade frequency
# - Target: 20-50 trades/year to avoid fee drag on 4h timeframe