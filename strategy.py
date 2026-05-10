#!/usr/bin/env python3
# 12h_Donchian_Breakout_WeeklyTrend_Volume
# Hypothesis: Use 12h Donchian breakouts filtered by weekly trend and volume spikes.
# In bull markets, weekly trend is up; we take long breakouts. In bear markets, weekly trend is down; we take short breakouts.
# Volume confirmation ensures breakouts have conviction. This reduces false breakouts in choppy markets.
# Target: 15-30 trades/year to stay within optimal trade frequency for 12h.

name = "12h_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "12h"
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
    volume = prices['volume'].values
    
    # Donchian channel on 12h: 20-period high/low
    donch_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donch_period, n):
        upper[i] = np.max(high[i-donch_period:i])
        lower[i] = np.min(low[i-donch_period:i])
    
    # Weekly trend filter: EMA 50 on 1w
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.5 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, donch_period)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian, weekly trend up, volume confirmation
            if close[i] > upper[i] and ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, weekly trend down, volume confirmation
            elif close[i] < lower[i] and ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below lower Donchian
            if close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above upper Donchian
            if close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals