#!/usr/bin/env python3
# 4h_4H_Aroon_Downtrend_Short_Only
# Hypothesis: In bear markets, Aroon Down > 70 on 4h signals strong downtrends.
# Combine with 1d EMA50 trend filter to avoid counter-trend shorts.
# Volume confirmation ensures momentum behind the move.
# Short-only strategy works in both bull (selective shorts in pullbacks) and bear (strong trends).
# Target: 20-40 trades/year to minimize fee drag.

name = "4h_4H_Aroon_Downtrend_Short_Only"
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
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Aroon Down (25-period) on 4h
    period = 25
    aroon_down = np.full(n, np.nan)
    for i in range(period - 1, n):
        window_low = low[i - period + 1:i + 1]
        lowest_low_idx = np.argmin(window_low)
        periods_since_low = period - 1 - lowest_low_idx
        aroon_down[i] = ((period - periods_since_low) / period) * 100
    
    # Align 1d trend to 4h
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume confirmation: 6-period (1.5-day) average on 4h
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(aroon_down[i]) or 
            np.isnan(trend_1d_down_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter short: Aroon Down > 70 + 1d downtrend + volume
            if (aroon_down[i] > 70 and 
                trend_1d_down_aligned[i] > 0.5 and 
                volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == -1:
            # Exit when Aroon Down < 30 or trend changes
            if (aroon_down[i] < 30 or 
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals