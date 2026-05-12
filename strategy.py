#!/usr/bin/env python3
# 4H_DONCHIAN_BREAKOUT_VOLUME_TREND
# Hypothesis: 4h Donchian(20) breakouts with 12h EMA50 trend filter and volume confirmation work in both bull and bear markets by capturing institutional breakouts while avoiding false signals in low-volume environments. The 12h EMA50 provides robust trend filtering that adapts to longer-term market direction, reducing whipsaws during choppy periods. Volume confirmation ensures breakouts have institutional participation. Designed for 20-40 trades/year to minimize fee drag.

name = "4H_DONCHIAN_BREAKOUT_VOLUME_TREND"
timeframe = "4h"
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
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > vol_ma * 1.5
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    pclose_12h = df_12h['close'].values
    ema50_12h = pd.Series(pclose_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(ema50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper band with volume confirmation and uptrend
            if close[i] > highest_high[i] and volume_confirm[i] and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower band with volume confirmation and downtrend
            elif close[i] < lowest_low[i] and volume_confirm[i] and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to Donchian middle or trend breaks
            middle = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < middle or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to Donchian middle or trend breaks
            middle = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > middle or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals