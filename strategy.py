#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla Pivot R3/S3 Breakout with 1-day Volume Spike and Choppiness Filter.
Long when price breaks above R3 with volume spike and choppiness > 61.8 (range).
Short when price breaks below S3 with volume spike and choppiness > 61.8.
Exit when price returns to H4/L4 levels.
Camarilla levels provide institutional support/resistance; volume confirms breakout strength;
choppiness filter ensures we only trade in ranging markets where mean reversion works.
Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by fading extremes in ranging conditions.
"""

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
    
    # Calculate daily Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for today's Camarilla levels
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # Camarilla formulas
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    R4 = prev_close + (prev_high - prev_low) * 1.1
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    S4 = prev_close - (prev_high - prev_low) * 1.1
    H4 = prev_close + (prev_high - prev_low) * 1.1 / 4
    L4 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume confirmation: current volume > 2.0x 24-period average (1 day of 4h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Choppiness Index: 4h period
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close_4h = np.roll(close, 1)
    prev_close_4h[0] = close[0]
    tr = true_range(high, low, prev_close_4h)
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_raw = 100 * np.log10(atr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop = np.where((highest_high_14 - lowest_low_14) > 0, chop_raw, 50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(vol_ma_24[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_24[i]
        
        # Choppiness filter: only trade in ranging markets (chop > 61.8)
        in_range = chop[i] > 61.8
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and in range
            if close[i] > R3_aligned[i] and vol_spike and in_range:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike and in range
            elif close[i] < S3_aligned[i] and vol_spike and in_range:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to H4/L4 levels (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to H4 or below
                if close[i] <= H4_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to L4 or above
                if close[i] >= L4_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_1dVol_Chop"
timeframe = "4h"
leverage = 1.0