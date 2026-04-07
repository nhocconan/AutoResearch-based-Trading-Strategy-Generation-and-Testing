#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d Trend Filter and Volume Confirmation
Long when price breaks above Donchian(20) high + price > 1d EMA50 (uptrend)
Short when price breaks below Donchian(20) low + price < 1d EMA50 (downtrend)
Exit when price crosses back through Donchian midpoint
Designed to capture trends with clear entry/exit rules
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channel (20) ===
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # === 1d EMA50 Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === Volume Confirmation (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian midpoint
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian midpoint
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 20-period average
            vol_ok = volume[i] > vol_ma[i]
            
            if vol_ok:
                # Long: break above Donchian high + uptrend (price > 1d EMA50)
                if close[i] > high_20[i] and close[i] > ema_50_aligned[i]:
                    position = 1
                    signals[i] = 0.30
                # Short: break below Donchian low + downtrend (price < 1d EMA50)
                elif close[i] < low_20[i] and close[i] < ema_50_aligned[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals