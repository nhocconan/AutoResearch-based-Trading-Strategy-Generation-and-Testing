#!/usr/bin/env python3
"""
4h_12h_camarilla_breakout
Uses 12h Camarilla pivot levels to identify key support/resistance. Breakouts occur when price moves beyond S3/R3 levels with volume confirmation on 4h. 
Trades only in low volatility regimes (Choppiness Index > 61.8) to avoid whipsaws. 
Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag.
Works in trending markets by capturing breakouts from key levels.
"""

name = "4h_12h_camarilla_breakout"
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
    
    # Get 12h data for Camarilla pivot levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate previous period's Camarilla levels
    # Using previous day's high, low, close for intraday calculation
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = high_12h[0]  # First value
    prev_low[0] = low_12h[0]
    prev_close[0] = close_12h[0]
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    # Avoid division by zero
    range_[range_ == 0] = 1e-10
    
    # Camarilla levels
    R4 = prev_close + range_ * 1.5
    R3 = prev_close + range_ * 1.25
    R2 = prev_close + range_ * 1.166
    R1 = prev_close + range_ * 1.083
    S1 = prev_close - range_ * 1.083
    S2 = prev_close - range_ * 1.166
    S3 = prev_close - range_ * 1.25
    S4 = prev_close - range_ * 1.5
    
    # Use S3 and R3 as breakout levels
    breakout_upper = R3
    breakout_lower = S3
    
    # Align Camarilla levels to 4h
    breakout_upper_aligned = align_htf_to_ltf(prices, df_12h, breakout_upper)
    breakout_lower_aligned = align_htf_to_ltf(prices, df_12h, breakout_lower)
    
    # Choppiness Index on 12h for regime filtering
    # Chop = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    range_max_min = max_high - min_low
    range_max_min[range_max_min == 0] = 1e-10
    
    chop = 100 * np.log10(atr_sum / range_max_min) / np.log10(14)
    chop[np.isnan(chop)] = 50  # Default value for insufficient data
    
    # Chop > 61.8 indicates ranging market (good for mean reversion, but we use it to avoid strong trends)
    # Actually, Chop > 61.8 = ranging, Chop < 38.2 = trending
    # We want to avoid choppy markets for breakouts, so we require Chop < 61.8 (trending or mildly ranging)
    chop_filter = chop < 61.8
    
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_filter)
    
    # Volume confirmation on 4h: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(breakout_upper_aligned[i]) or np.isnan(breakout_lower_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above R3 with volume, in non-choppy regime
        if breakout_upper_aligned[i] > 0 and close[i] > breakout_upper_aligned[i] and vol_confirm[i] and chop_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below S3 with volume, in non-choppy regime
        elif breakout_lower_aligned[i] > 0 and close[i] < breakout_lower_aligned[i] and vol_confirm[i] and chop_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: return to midpoint of previous day's range
        elif position == 1 and close[i] <= (breakout_upper_aligned[i] + breakout_lower_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= (breakout_upper_aligned[i] + breakout_lower_aligned[i]) / 2:
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