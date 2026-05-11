#!/usr/bin/env python3
# 6h_WeeklyPivot_Trend_VolumeBreakout_v1
# Hypothesis: Breakout of weekly pivot levels (R1/S1) with 1-week EMA50 trend filter and volume confirmation.
# Uses 6h timeframe with weekly HTF for trend direction, targeting 15-30 trades/year.
# Weekly pivot provides strong structural levels; EMA50 filters trend; volume confirms breakout strength.
# Designed to work in both bull and bear markets by aligning with higher-timeframe trend.

name = "6h_WeeklyPivot_Trend_VolumeBreakout_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # === Weekly Data (loaded ONCE) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === Weekly Pivot Levels (R1, S1) ===
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r1 = pivot_1w + (range_1w * 1.1 / 4)
    s1 = pivot_1w - (range_1w * 1.1 / 4)
    
    # Align weekly levels to 6h
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    
    # === Weekly EMA50 Trend Filter ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_6h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Volume Spike Filter (24-period EMA) ===
    vol_ema24 = pd.Series(volume).ewm(span=24, min_periods=24, adjust=False).mean().values
    volume_ok = volume > vol_ema24 * 1.5  # Require 1.5x average volume
    
    # === Signal Parameters ===
    position_size = 0.25  # 25% of capital per trade
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    holding_bars = 0
    
    # Start after warmup (covers EMA50)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(ema50_1w_6h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                holding_bars = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above R1 (both open and close) + above weekly EMA50 + volume spike
            if (open_price[i] > r1_6h[i] and close[i] > r1_6h[i] and 
                close[i] > ema50_1w_6h[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
                holding_bars = 0
            # Short: Break below S1 (both open and close) + below weekly EMA50 + volume spike
            elif (open_price[i] < s1_6h[i] and close[i] < s1_6h[i] and 
                  close[i] < ema50_1w_6h[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
                holding_bars = 0
        else:
            # Enforce minimum holding period (6 bars = 1 day)
            holding_bars += 1
            if holding_bars < 6:
                signals[i] = position_size if position == 1 else -position_size
                continue
            
            # Exit: Price closes below/above opposite level
            if position == 1:
                if close[i] < s1_6h[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > r1_6h[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = -position_size
    
    return signals