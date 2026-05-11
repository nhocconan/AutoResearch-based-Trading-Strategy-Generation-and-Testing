#!/usr/bin/env python3
# 1d_WeeklyPivot_Breakout_Trend_Filter
# Hypothesis: Breakout of weekly (1w) Camarilla R1/S1 levels with 1-day EMA50 trend filter and volume confirmation.
# Designed for low-frequency, high-conviction trades on 1d timeframe to survive bear markets (2025+).
# Weekly pivots provide stronger structural levels; EMA50 filters counter-trend noise; volume ensures conviction.
# Targets 10-20 trades/year (~40-80 total over 4 years) to minimize fee drag.

name = "1d_WeeklyPivot_Breakout_Trend_Filter"
timeframe = "1d"
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
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # === Weekly (1w) Data (loaded ONCE) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === Weekly Camarilla Pivot Levels (R1, S1) ===
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r1_1w = pivot_1w + (range_1w * 1.1 / 12)
    s1_1w = pivot_1w - (range_1w * 1.1 / 12)
    
    # Align weekly levels to daily
    r1_1d = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1d = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === Daily EMA50 Trend Filter ===
    ema50_1d = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # Require 1.5x average volume
    
    # === Signal Parameters ===
    position_size = 0.25  # 25% of capital per trade
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    holding_bars = 0
    
    # Start after warmup (covers EMA50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or 
            np.isnan(ema50_1d[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                holding_bars = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above R1 (both open and close) + above EMA50 + volume spike
            if (open_price[i] > r1_1d[i] and close[i] > r1_1d[i] and 
                close[i] > ema50_1d[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
                holding_bars = 0
            # Short: Break below S1 (both open and close) + below EMA50 + volume spike
            elif (open_price[i] < s1_1d[i] and close[i] < s1_1d[i] and 
                  close[i] < ema50_1d[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
                holding_bars = 0
        else:
            # Enforce minimum holding period (3 days)
            holding_bars += 1
            if holding_bars < 3:
                signals[i] = position_size if position == 1 else -position_size
                continue
            
            # Exit: Price closes below/above opposite level
            if position == 1:
                if close[i] < s1_1d[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > r1_1d[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = -position_size
    
    return signals