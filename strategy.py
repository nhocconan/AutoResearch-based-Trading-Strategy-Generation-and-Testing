#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Targets 20-50 trades per year (75-200 total over 4 years) to minimize fee drag
# Donchian channels provide robust breakout levels from recent price extremes
# 12h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades
# Volume confirmation (1.8x 20-period average) filters false breakouts
# Uses discrete position sizing 0.28 to balance exposure and risk
# Works in both bull and bear: trend filter prevents counter-trend entries during reversals

name = "4h_Donchian20_12hEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA50
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 4h volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper AND above 12h EMA50 AND volume confirm
            if (close[i] > high_ma[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.28
                position = 1
            # Short: price breaks below Donchian lower AND below 12h EMA50 AND volume confirm
            elif (close[i] < low_ma[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.28
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian lower OR below 12h EMA50
            if (close[i] < low_ma[i] or 
                close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper OR above 12h EMA50
            if (close[i] > high_ma[i] or 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals