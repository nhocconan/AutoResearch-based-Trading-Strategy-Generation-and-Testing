#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla R1/S1 breakout on 12h with 1-day trend filter and volume spike.
Works in both bull and bear by using 1d EMA50 to filter trade direction (long in uptrend, short in downtrend).
Designed for low trade frequency (~25/year) to minimize fee drag on 12h timeframe.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Calculate Camarilla levels from prior 1d bar ===
    # Get daily OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC (shifted by 1 to avoid look-ahead)
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Align to 12h timeframe
    prior_high_12h = align_htf_to_ltf(prices, df_1d, prior_high)
    prior_low_12h = align_htf_to_ltf(prices, df_1d, prior_low)
    prior_close_12h = align_htf_to_ltf(prices, df_1d, prior_close)
    
    # Camarilla R1 and S1 levels
    R1 = prior_close_12h + (prior_high_12h - prior_low_12h) * 1.1 / 12
    S1 = prior_close_12h - (prior_high_12h - prior_low_12h) * 1.1 / 12
    
    # === 1-day EMA50 Trend Filter ===
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Volume Spike Filter (2x 20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 2.0
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers EMA and Camarilla calculation)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(ema50_1d_12h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close crosses above R1 with uptrend (close > EMA50) and volume spike
            if (close[i] > R1[i] and 
                close[i] > ema50_1d_12h[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: Close crosses below S1 with downtrend (close < EMA50) and volume spike
            elif (close[i] < S1[i] and 
                  close[i] < ema50_1d_12h[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Close crosses back through the Camarilla level in opposite direction
            if position == 1:
                if close[i] < S1[i]:  # Exit long if price breaks below S1
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > R1[i]:  # Exit short if price breaks above R1
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals