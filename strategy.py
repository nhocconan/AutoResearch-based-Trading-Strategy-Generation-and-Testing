#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) Breakout with 1w EMA50 Trend Filter and Volume Spike
- Uses 1d Donchian breakout (price > highest high of last 20 days or < lowest low) 
- Trend filter: 1w EMA50 (bullish if price > EMA50, bearish if price < EMA50)
- Volume confirmation: current volume > 2.0x 20-day average volume
- Designed for 1d timeframe to minimize trade frequency and avoid fee drag
- Target: 7-25 trades/year per symbol (30-100 total over 4 years)
- Works in both bull and bear markets via 1w EMA50 trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper: highest high of last 20 days
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian lower: lowest low of last 20 days
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: > 2.0x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50_1w and Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > Donchian Upper (breakout) AND price > 1w EMA50 (uptrend) AND volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < Donchian Lower (breakdown) AND price < 1w EMA50 (downtrend) AND volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Close back inside Donchian channel OR loss of trend
            exit_signal = False
            if position == 1:
                # Exit long when close < Donchian Lower OR price < 1w EMA50
                if close[i] < donchian_low_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when close > Donchian Upper OR price > 1w EMA50
                if close[i] > donchian_high_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0