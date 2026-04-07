#!/usr/bin/env python3
"""
4h Donchian Breakout with 12h Trend and Volume Confirmation
Long when price breaks above Donchian(20) high and 12h EMA50 > EMA200 + volume spike
Short when price breaks below Donchian(20) low and 12h EMA50 < EMA200 + volume spike
Exit when price crosses Donchian midline (10-period average)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
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
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2
    
    # === 12h EMA Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False).mean().values
    ema_200_12h = pd.Series(df_12h['close'].values).ewm(span=200, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # === Volume Spike (2x 20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_200_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian midline
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian midline
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Bullish trend: 12h EMA50 > EMA200
            if ema_50_12h_aligned[i] > ema_200_12h_aligned[i]:
                # Long entry: price breaks above Donchian high + volume spike
                if close[i] > highest_20[i] and volume_spike[i]:
                    position = 1
                    signals[i] = 0.25
            # Bearish trend: 12h EMA50 < EMA200
            elif ema_50_12h_aligned[i] < ema_200_12h_aligned[i]:
                # Short entry: price breaks below Donchian low + volume spike
                if close[i] < lowest_20[i] and volume_spike[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals