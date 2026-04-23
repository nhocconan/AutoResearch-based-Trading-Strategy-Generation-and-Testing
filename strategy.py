#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above upper Donchian channel in 12h uptrend with volume > 1.8x 20-period MA.
Short when price breaks below lower Donchian channel in 12h downtrend with volume > 1.8x 20-period MA.
Exit when price touches the opposite Donchian channel or 12h EMA50.
Uses 12h HTF for trend alignment to reduce whipsaw. Designed for ~20-40 trades/year with strong edge in both bull and bear markets via trend filter and volatility-based channels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h Donchian channels (20-period)
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # need EMA50, Donchian20, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(high_roll_max[i]) or 
            np.isnan(low_roll_min[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 12h close > EMA50 = uptrend, close < EMA50 = downtrend
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        trend_up = close_12h_aligned[i] > ema_50_12h_aligned[i]
        trend_down = close_12h_aligned[i] < ema_50_12h_aligned[i]
        
        # Volume filter: 4h volume > 1.8x 20-period MA (strong confirmation)
        vol_filter = volume[i] > 1.8 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_upper = close[i] > high_roll_max[i]
        breakdown_lower = close[i] < low_roll_min[i]
        touch_upper = abs(close[i] - high_roll_max[i]) < (high_roll_max[i] * 0.001)  # within 0.1% of upper channel
        touch_lower = abs(close[i] - low_roll_min[i]) < (low_roll_min[i] * 0.001)   # within 0.1% of lower channel
        touch_ema = abs(close[i] - ema_50_12h_aligned[i]) < (ema_50_12h_aligned[i] * 0.001)  # within 0.1% of EMA
        
        if position == 0:
            # Long: Price breaks above upper Donchian AND uptrend AND volume spike
            if breakout_upper and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian AND downtrend AND volume spike
            elif breakdown_lower and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Price touches opposite channel or returns to EMA50
            exit_signal = False
            
            if position == 1:
                # Long exit: Price touches lower Donchian or returns to EMA50
                if touch_lower or touch_ema:
                    exit_signal = True
            elif position == -1:
                # Short exit: Price touches upper Donchian or returns to EMA50
                if touch_upper or touch_ema:
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