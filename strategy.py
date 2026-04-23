#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Uses 1d Donchian channels for breakout detection, 1w EMA50 for primary trend filter,
and volume spike for momentum confirmation. Designed for 1d timeframe to minimize
fee drag while capturing major trend moves. Target: 15-30 trades/year per symbol (60-120 total over 4 years).
Uses discrete position sizing (0.25) to balance return and fee drag.
Works in both bull and bear markets by following the 1w EMA50 trend direction.
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
    
    # Calculate 1d Donchian channels (20-period)
    if n < 20:
        return np.zeros(n)
    
    # Calculate rolling max/min for Donchian channels
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # need Donchian20, EMA50, and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_rolling_max[i]) or np.isnan(low_rolling_min[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1w EMA50
        trend_up = close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Break above Donchian upper band AND uptrend on 1w AND volume spike
            if close[i] > high_rolling_max[i] and trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower band AND downtrend on 1w AND volume spike
            elif close[i] < low_rolling_min[i] and trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: break of opposite Donchian band (lower band for longs, upper band for shorts)
            exit_signal = False
            if position == 1:
                # Exit long on break below Donchian lower band
                if close[i] < low_rolling_min[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Donchian upper band
                if close[i] > high_rolling_max[i]:
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