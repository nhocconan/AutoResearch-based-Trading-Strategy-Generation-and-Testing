#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_volume_regime_v3
Hypothesis: Combine Camarilla breakouts with volume confirmation and a 1d Choppiness regime filter.
In chop (CHOP > 61.8), mean-revert at S3/R3; in trend (CHOP < 38.2), breakout at S4/R4.
Uses 1d data for context, 4h for execution. Target: 20-50 trades/year.
"""

name = "4h_1d_camarilla_breakout_volume_regime_v3"
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
    
    # Get daily data for context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Calculate true range for ATR (used in Choppiness)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (14-day)
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(14)
    
    # Previous day's range for Camarilla levels
    prev_range = prev_high - prev_low
    # Resistance levels
    r3 = prev_close + prev_range * 1.1 / 2
    r4 = prev_close + prev_range * 1.1
    # Support levels
    s3 = prev_close - prev_range * 1.1 / 2
    s4 = prev_close - prev_range * 1.1
    
    # Align all 1d data to 4h
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
        
        chop_val = chop_aligned[i]
        
        if chop_val > 61.8:  # Choppy market - mean reversion
            # Long at S3, exit at R3
            if close[i] <= s3_aligned[i] and vol_confirm[i] and position != 1:
                position = 1
                signals[i] = 0.25
            elif close[i] >= r3_aligned[i] and position == 1:
                position = 0
                signals[i] = 0.0
            # Short at R3, exit at S3
            elif close[i] >= r3_aligned[i] and vol_confirm[i] and position != -1:
                position = -1
                signals[i] = -0.25
            elif close[i] <= s3_aligned[i] and position == -1:
                position = 0
                signals[i] = 0.0
            else:
                # Hold
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:  # Trending market - breakout
            # Long breakout above R4
            if close[i] > r4_aligned[i] and vol_confirm[i] and position != 1:
                position = 1
                signals[i] = 0.25
            # Short breakdown below S4
            elif close[i] < s4_aligned[i] and vol_confirm[i] and position != -1:
                position = -1
                signals[i] = -0.25
            # Exit on opposite touch
            elif position == 1 and close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                # Hold
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
    
    return signals