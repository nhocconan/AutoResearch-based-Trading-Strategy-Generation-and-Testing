#!/usr/bin/env python3
# 12h_1W_Trend_1D_Volume_Spike
# Hypothesis: Use 1-week trend direction and 1-day volume spike for entry on 12h timeframe.
# Long when weekly close > weekly EMA50 and daily volume > 2x 20-day average; enter on 12h pullback to EMA20.
# Short when weekly close < weekly EMA50 and daily volume > 2x 20-day average; enter on 12h bounce to EMA20.
# Designed for low trade frequency (12-37/year) to avoid fee drag, works in bull/bear via trend filter.

name = "12h_1W_Trend_1D_Volume_Spike"
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
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1w EMA50 trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_down = close_1w < ema50_1w
    
    # Align 1w trend to 12h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # 1d volume spike filter: current volume > 2 * 20-day average
    volume_1d = df_1d['volume'].values
    volume_series_1d = pd.Series(volume_1d)
    vol_ma_20_1d = volume_series_1d.rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2 * vol_ma_20_1d)
    
    # Align 1d volume spike to 12h
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # 12h EMA20 for entry timing
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1w uptrend + volume spike + price near EMA20 with pullback
            if (trend_1w_up_aligned[i] > 0.5 and 
                vol_spike_1d_aligned[i] > 0.5 and
                close[i] <= ema20[i] * 1.01 and  # within 1% above EMA20 (pullback)
                close[i] >= ema20[i] * 0.99):    # within 1% below EMA20
                signals[i] = 0.25
                position = 1
            # Short: 1w downtrend + volume spike + price near EMA20 with bounce
            elif (trend_1w_down_aligned[i] > 0.5 and 
                  vol_spike_1d_aligned[i] > 0.5 and
                  close[i] >= ema20[i] * 0.99 and  # within 1% below EMA20
                  close[i] <= ema20[i] * 1.01):    # within 1% above EMA20
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: 1w trend fails or volume spike ends
            if (trend_1w_up_aligned[i] < 0.5 or 
                vol_spike_1d_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: 1w trend fails or volume spike ends
            if (trend_1w_down_aligned[i] < 0.5 or 
                vol_spike_1d_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals