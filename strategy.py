#!/usr/bin/env python3
# 6h_1d_elliot_wave_oscillator_v1
# Strategy: 6h Elliott Wave Oscillator with 1d EMA trend filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: The Elliott Wave Oscillator (difference between 34-period and 5-period SMA) captures momentum shifts.
# Combined with 1d EMA trend filter and volume confirmation, it avoids false signals and captures sustained moves.
# Designed for low trade frequency (~15-30/year) to avoid fee drag in 6h timeframe.
# Works in both bull and bear markets by following the higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elliot_wave_oscillator_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate Elliott Wave Oscillator (34 SMA - 5 SMA) on 1d
    close_1d = df_1d['close'].values
    sma_5_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    sma_34_1d = pd.Series(close_1d).rolling(window=34, min_periods=34).mean().values
    ewo_1d = sma_34_1d - sma_5_1d  # Elliott Wave Oscillator
    
    # Align EWO to 6h timeframe
    ewo_1d_aligned = align_htf_to_ltf(prices, df_1d, ewo_1d)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(ewo_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # EWO signals: zero-line cross
        ewo_cross_up = ewo_1d_aligned[i] > 0 and ewo_1d_aligned[i-1] <= 0
        ewo_cross_down = ewo_1d_aligned[i] < 0 and ewo_1d_aligned[i-1] >= 0
        
        # 1d EMA trend filter: price above EMA = bullish trend, below = bearish
        trend_bullish = close[i] > ema_50_1d_aligned[i]
        trend_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: EWO crosses above zero AND bullish trend AND volume confirmation
        if ewo_cross_up and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: EWO crosses below zero AND bearish trend AND volume confirmation
        elif ewo_cross_down and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite EWO cross
        elif position == 1 and ewo_cross_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and ewo_cross_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals