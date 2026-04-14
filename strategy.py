#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d 144-day exponential moving average (EMA) as trend filter,
# combined with 1d Donchian(20) breakout and volume confirmation.
# Long when price > 1d EMA144 AND breaks above 1d Donchian upper band AND 4h volume > 1.5x 20-period average.
# Short when price < 1d EMA144 AND breaks below 1d Donchian lower band AND 4h volume > 1.5x 20-period average.
# Exit when price crosses the 1d EMA144 in the opposite direction.
# The 144-day EMA provides a strong trend filter that works in both bull and bear markets,
# while Donchian breakouts capture momentum and volume confirmation reduces false signals.
# Target: 20-30 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 144:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA and Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 144:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 144-period EMA on 1d close
    ema_144 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 144:
        # Calculate initial SMA for EMA seed
        ema_144[143] = np.mean(close_1d[:144])
        # Calculate EMA using Wilder's smoothing (alpha = 2/(N+1))
        alpha = 2.0 / (144 + 1)
        for i in range(144, len(close_1d)):
            ema_144[i] = alpha * close_1d[i] + (1 - alpha) * ema_144[i-1]
    
    # Calculate 20-period Donchian channel on 1d data
    donchian_up = np.full_like(high_1d, np.nan)
    donchian_down = np.full_like(low_1d, np.nan)
    for i in range(19, len(high_1d)):
        donchian_up[i] = np.max(high_1d[i-19:i+1])
        donchian_down[i] = np.min(low_1d[i-19:i+1])
    
    # Load 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    
    # Calculate 20-period average volume on 4h data
    vol_ma_20 = np.full_like(volume_4h, np.nan)
    for i in range(19, len(volume_4h)):
        vol_ma_20[i] = np.mean(volume_4h[i-19:i+1])
    
    # Align indicators to 4h timeframe
    ema_144_aligned = align_htf_to_ltf(prices, df_1d, ema_144)
    donchian_up_aligned = align_htf_to_ltf(prices, df_1d, donchian_up)
    donchian_down_aligned = align_htf_to_ltf(prices, df_1d, donchian_down)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(144, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_144_aligned[i]) or 
            np.isnan(donchian_up_aligned[i]) or 
            np.isnan(donchian_down_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 20-period average
        volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
        volume_ratio = volume_4h_aligned[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Look for entries: EMA filter + Donchian breakout + volume confirmation
            # Long: price > EMA144 AND break above upper band AND volume > 1.5x average
            if (close[i] > ema_144_aligned[i] and 
                close[i] > donchian_up_aligned[i] and
                volume_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Short: price < EMA144 AND break below lower band AND volume > 1.5x average
            elif (close[i] < ema_144_aligned[i] and 
                  close[i] < donchian_down_aligned[i] and
                  volume_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA144
            if close[i] < ema_144_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above EMA144
            if close[i] > ema_144_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_EMA144_Donchian_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0