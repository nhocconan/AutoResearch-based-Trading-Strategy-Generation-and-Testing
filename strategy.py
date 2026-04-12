#!/usr/bin/env python3
"""
12h_1d_Donchian_Breakout_Volume_Trend
Hypothesis: On 12h timeframe, enter long when price breaks above Donchian(20) high and volume spikes, 
enter short when price breaks below Donchian(20) low and volume spikes. Uses 1d for trend direction 
(1d EMA50) to filter trades. Volume confirmation ensures breakout strength. Works in bull (breakouts 
in uptrend) and bear (breakdowns in downtrend). Target: 15-25 trades per year per symbol (60-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Donchian_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12H INDICATORS: Donchian Channel (20) and Volume MA (20) ===
    # Donchian high: rolling max of high
    donchian_high = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
    
    # Donchian low: rolling min of low
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume MA(20)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # === 1D INDICATOR: EMA(50) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter from 1d EMA50
        uptrend_1d = close[i] > ema_50_1d_aligned[i]
        downtrend_1d = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i] and volume_spike
        short_breakout = close[i] < donchian_low[i] and volume_spike
        
        # Exit conditions: trend reversal or opposite breakout
        exit_long = not uptrend_1d or short_breakout
        exit_short = not downtrend_1d or long_breakout
        
        if long_breakout and uptrend_1d and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and downtrend_1d and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals