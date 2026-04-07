#!/usr/bin/env python3
"""
1d Donchian Breakout with 1w Trend Filter and Volume Confirmation
Long when price breaks above Donchian(20) high in uptrend (1w EMA50 > EMA200) with volume > 1.5x average
Short when price breaks below Donchian(20) low in downtrend (1w EMA50 < EMA200) with volume > 1.5x average
Exit when price crosses Donchian midpoint or trend reverses
Designed to capture trends with filtered breakouts in both bull and bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
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
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # === Volume Average (20) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1w EMA Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(df_1w['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below midpoint OR trend reverses
            if close[i] < donchian_mid[i] or ema_50_aligned[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses above midpoint OR trend reverses
            if close[i] > donchian_mid[i] or ema_50_aligned[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Volume confirmation: volume > 1.5x average
            vol_confirm = volume[i] > 1.5 * vol_ma[i]
            
            # Long: breakout above Donchian high in uptrend
            if close[i] > highest_high[i] and ema_50_aligned[i] > ema_200_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.30
            # Short: breakdown below Donchian low in downtrend
            elif close[i] < lowest_low[i] and ema_50_aligned[i] < ema_200_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.30
    
    return signals