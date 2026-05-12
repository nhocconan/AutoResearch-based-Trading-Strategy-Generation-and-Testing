#!/usr/bin/env python3
"""
6h Weekly Pivot + Daily Trend + Volume Spike
Hypothesis: Weekly pivot levels (R1/S1) act as strong support/resistance. 
In trending markets (above/below daily EMA34), price tends to break these levels with volume.
Fade at R1/S1 in ranging markets, breakout continuation in trending markets.
Uses weekly pivot for structure, daily EMA for trend filter, volume for confirmation.
Designed for low trade frequency (15-25/year) to minimize fee drag while capturing sustained moves.
"""
name = "6h_WeeklyPivot_DailyTrend_Volume"
timeframe = "6h"
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
    
    # === Weekly Pivot Points (calculate once, update weekly) ===
    df_1w = get_htf_data(prices, '1w')
    # Typical price for weekly pivot
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_pivot = typical_price
    weekly_r1 = 2 * weekly_pivot - df_1w['low']
    weekly_s1 = 2 * weekly_pivot - df_1w['high']
    # Align to 6h timeframe (wait for weekly bar to close)
    pivot_w = align_htf_to_ltf(prices, df_1w, weekly_pivot.values)
    r1_w = align_htf_to_ltf(prices, df_1w, weekly_r1.values)
    s1_w = align_htf_to_ltf(prices, df_1w, weekly_s1.values)
    
    # === Daily EMA34 (trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # === Volume Spike (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Require 2x average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_w[i]) or np.isnan(r1_w[i]) or np.isnan(s1_w[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above S1, above daily EMA (uptrend), with volume spike
            # OR price breaks above R1 with volume (breakout)
            if ((close[i] > s1_w[i] and close[i] > ema_34_aligned[i] and vol_spike[i]) or
                (close[i] > r1_w[i] and vol_spike[i])):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below R1, below daily EMA (downtrend), with volume spike
            # OR price breaks below S1 with volume (breakdown)
            elif ((close[i] < r1_w[i] and close[i] < ema_34_aligned[i] and vol_spike[i]) or
                  (close[i] < s1_w[i] and vol_spike[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price below S1 OR below daily EMA (trend change)
            if close[i] < s1_w[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above R1 OR above daily EMA (trend change)
            if close[i] > r1_w[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals