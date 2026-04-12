#!/usr/bin/env python3
# 12h_1d_camarilla_breakout_volume_regime
# Hypothesis: 12-hour Camarilla breakout with volume confirmation and chop regime filter.
# Uses 1d Camarilla levels for structure, volume > 1.5x 20-period average for confirmation,
# and Choppiness Index > 61.8 to filter for ranging markets where mean reversion at S3/R3 works.
# Designed to work in both bull and bear by avoiding trending markets (CHOP < 38.2) and
# only taking mean-reversion trades in chop. Target: 12-37 trades/year (50-150 total over 4 years).

name = "12h_1d_camarilla_breakout_volume_regime"
timeframe = "12h"
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
    
    # Get daily data for Camarilla and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range for Camarilla
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Calculate previous day's range (handle first value)
    range_ = prev_high - prev_low
    range_[0] = 0  # First day has no previous day
    
    # Camarilla levels (based on previous day)
    r3 = prev_close + range_ * 1.1 / 2
    r4 = prev_close + range_ * 1.1
    s3 = prev_close - range_ * 1.1 / 2
    s4 = prev_close - range_ * 1.1
    
    # Choppiness Index (14-period)
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = true_range(high_1d, low_1d, np.roll(close_1d, 1))
    tr[0] = high_1d[0] - low_1d[0]  # First TR
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = highest_high_14 - lowest_low_14
    range_14[range_14 == 0] = 1e-10
    
    chop = 100 * np.log10(atr_14 / range_14) / np.log10(14)
    chop[np.isnan(chop)] = 50  # Default to middle when not enough data
    
    # Align Camarilla levels and chop to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in ranging markets (chop > 61.8)
        if chop_aligned[i] <= 61.8:
            # In trending markets, stay flat
            if position == 1:
                signals[i] = 0.0
                position = 0
            elif position == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Mean reversion in chop: sell at R4, buy at S4
        if (close[i] > r4_aligned[i] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        elif (close[i] < s4_aligned[i] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Exit when price reaches opposite S3/R3 levels
        elif position == 1 and close[i] >= r3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= s3_aligned[i]:
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