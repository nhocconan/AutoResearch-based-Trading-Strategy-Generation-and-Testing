#!/usr/bin/env python3

# 4h_1d_trix_volume_regime
# Hypothesis: 4-hour TRIX momentum with volume spike confirmation and choppiness regime filter
# Works in bull/bear by using TRIX for momentum direction, volume spikes for institutional interest,
# and choppiness filter to avoid ranging markets. Target: 25-50 trades/year (100-200 total) to minimize fee drag.

name = "4h_1d_trix_volume_regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for TRIX and choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # TRIX calculation (15-period triple EMA)
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix_values = trix.fillna(0).values
    
    # Choppiness Index (14-period) - range detection
    atr_period = 14
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum()
    
    highest_high = pd.Series(high_1d).rolling(window=atr_period, min_periods=atr_period).max()
    lowest_low = pd.Series(low_1d).rolling(window=atr_period, min_periods=atr_period).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(atr_period)
    chop_values = chop.fillna(50).values  # neutral when not enough data
    
    # Volume spike detection (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # Align indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: TRIX positive + volume spike + trending market (CHOP < 38.2)
        if (trix_aligned[i] > 0 and vol_spike[i] and chop_aligned[i] < 38.2 and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: TRIX negative + volume spike + trending market (CHOP < 38.2)
        elif (trix_aligned[i] < 0 and vol_spike[i] and chop_aligned[i] < 38.2 and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: TRIX crosses zero OR choppy market detected (CHOP > 61.8)
        elif position == 1 and (trix_aligned[i] <= 0 or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (trix_aligned[i] >= 0 or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals