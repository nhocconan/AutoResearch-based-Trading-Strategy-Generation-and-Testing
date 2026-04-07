#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d Trend Filter and Volume Confirmation
Long when price breaks above 12h Donchian upper channel and 1d EMA50 > EMA200
Short when price breaks below 12h Donchian lower channel and 1d EMA50 < EMA200
Exit when price crosses back through the middle of the Donchian channel
Designed to capture trends with high-probability entries in both bull and bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # === Volume Filter (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_200 = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian middle
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian middle
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > 1.5 * vol_ma[i]
            
            if vol_confirm:
                # Bullish trend: EMA50 > EMA200
                if ema_50_aligned[i] > ema_200_aligned[i]:
                    # Long entry: break above Donchian upper channel
                    if close[i] > donchian_high[i]:
                        position = 1
                        signals[i] = 0.25
                # Bearish trend: EMA50 < EMA200
                elif ema_50_aligned[i] < ema_200_aligned[i]:
                    # Short entry: break below Donchian lower channel
                    if close[i] < donchian_low[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals