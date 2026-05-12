#!/usr/bin/env python3
"""
12h Donchian Breakout with Daily Trend and Volume
Hypothesis: Donchian breakouts on 12h capture medium-term trends, filtered by daily EMA trend and volume confirmation.
This reduces false signals in choppy markets while capturing sustained moves. Designed for low trade frequency (~20-50/year)
to minimize fee drift, suitable for both bull and bear markets via trend-following logic.
"""
name = "12h_Donchian_Breakout_DailyTrend_Volume"
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
    
    # === Daily EMA for Trend Filter (34) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 12h Donchian Channels (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume Spike (20) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high + above daily EMA + volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema_34_1d_aligned[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + below daily EMA + volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price below Donchian low OR below daily EMA
            if close[i] < lowest_low[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above Donchian high OR above daily EMA
            if close[i] > highest_high[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals