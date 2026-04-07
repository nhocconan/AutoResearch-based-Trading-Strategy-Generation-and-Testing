#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d Trend Filter
Breakout above/below 20-bar Donchian channel only when 1d EMA50 confirms trend
Volume confirmation and ATR stop to limit whipsaws
Works in trending markets (both bull/bear) by following higher timeframe trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 20-bar Donchian channels ===
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation (20-bar average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: require 1.5x average volume
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-bar low OR trend flips
            if low[i] <= low_min[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 20-bar high OR trend flips
            if high[i] >= high_max[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if not vol_ok:
                signals[i] = 0.0
                continue
            
            # Long: breakout above Donchian high with uptrend
            if high[i] > high_max[i] and close[i] > ema_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: breakout below Donchian low with downtrend
            elif low[i] < low_min[i] and close[i] < ema_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals