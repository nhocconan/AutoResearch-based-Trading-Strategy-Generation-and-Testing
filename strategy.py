#!/usr/bin/env python3
# 4h_Breakout_1dATR_Trend
# Hypothesis: Uses Donchian breakout on 4h with 1d ATR-based trend filter and volume confirmation.
# Long when: 4h price breaks above Donchian high (20), 1d ATR trend is up, and volume > 1.5x 20-period average.
# Short when: 4h price breaks below Donchian low (20), 1d ATR trend is down, and volume > 1.5x 20-period average.
# Exit when price returns to Donchian midline or ATR trend reverses.
# Designed to capture trends in both bull and bear markets with volatility-adjusted breakouts and volume confirmation.

name = "4h_Breakout_1dATR_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for ATR trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d ATR-based trend filter ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    
    # ATR(10)
    atr_period = 10
    atr = np.full(n, np.nan)
    for i in range(atr_period, len(tr)):
        atr[i] = np.mean(tr[i-atr_period:i+1])
    
    # ATR trend: rising if current ATR > previous ATR
    atr_up = np.zeros(len(atr), dtype=bool)
    atr_down = np.zeros(len(atr), dtype=bool)
    for i in range(1, len(atr)):
        if not np.isnan(atr[i]) and not np.isnan(atr[i-1]):
            atr_up[i] = atr[i] > atr[i-1]
            atr_down[i] = atr[i] < atr[i-1]
    
    # Align 1d ATR trend to 4h
    atr_up_aligned = align_htf_to_ltf(prices, df_1d, atr_up)
    atr_down_aligned = align_htf_to_ltf(prices, df_1d, atr_down)
    
    # --- 4h Donchian channels (20-period) ---
    donchian_period = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(donchian_period, n):
        highest_high[i] = np.max(high[i-donchian_period:i+1])
        lowest_low[i] = np.min(low[i-donchian_period:i+1])
    
    # Donchian midline
    mid = (highest_high + lowest_low) / 2.0
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Donchian(20), ATR(10), vol MA(20)
    start_idx = max(donchian_period, atr_period, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr_up_aligned[i]) or
            np.isnan(atr_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Volume spike
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            if breakout_up and atr_up_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            elif breakout_down and atr_down_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price returns to midline OR ATR trend turns down
                if close[i] < mid[i] or not atr_up_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to midline OR ATR trend turns up
                if close[i] > mid[i] or not atr_down_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals