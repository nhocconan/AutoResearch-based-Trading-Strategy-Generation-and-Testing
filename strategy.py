#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d Trend and Volume Filter
Long when price breaks above Donchian(20) high with 1d EMA(50) uptrend and volume spike
Short when price breaks below Donchian(20) low with 1d EMA(50) downtrend and volume spike
Exit when price breaks opposite Donchian band or trend reverses
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v2"
timeframe = "4h"
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
    
    # === Donchian Channel (20) ===
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 1d EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Volume Confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR 1d trend turns down
            if close[i] < donchian_low[i] or ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR 1d trend turns up
            if close[i] > donchian_high[i] or ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume spike (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with 1d trend confirmation
            if close[i] > donchian_high[i] and ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                # Price above Donchian high with uptrend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_low[i] and ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                # Price below Donchian low with downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals