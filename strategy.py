#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
Uses Donchian channels from 4h timeframe for entry, combined with
12h EMA50 to ensure trend alignment and avoid counter-trend entries.
Volume spike confirms breakout momentum. Designed for 4h timeframe to reduce trade frequency.
Works in both bull and bear markets by only trading in direction of 12h trend.
Target: 19-50 trades/year per symbol (75-200 total over 4 years).
Uses discrete position sizing (0.25) to minimize fee drag while maintaining profitability.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian channels: upper = rolling max(high, 20), lower = rolling min(low, 20)
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h timeframe (previous 20-bar completed values)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 50)  # need volume MA20, Donchian, and EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma_20[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian upper AND price > 12h EMA50 (uptrend) AND volume spike
            if close[i] > donchian_upper_aligned[i] and close[i] > ema_50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND price < 12h EMA50 (downtrend) AND volume spike
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema_50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: break of opposite Donchian level (lower for longs, upper for shorts)
            exit_signal = False
            if position == 1:
                # Exit long on break below Donchian lower
                if close[i] < donchian_lower_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Donchian upper
                if close[i] > donchian_upper_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0