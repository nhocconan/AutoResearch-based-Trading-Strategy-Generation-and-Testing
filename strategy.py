#!/usr/bin/env python3
"""
12h_Donchian_20_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Donchian(20) breakout on 12h with 1d EMA50 trend filter and volume spike confirmation. Works in bull markets (buy upper band breakouts in uptrend) and bear markets (sell lower band breakouts in downtrend). Volume spike confirms institutional interest. Target: 15-35 trades per year on 12h timeframe.
"""

name = "12h_Donchian_20_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

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
    
    # === 1D Data for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Trend filter: EMA50 on 1d close
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === 12H Donchian Channel (20-period) ===
    # Calculate Donchian bands on 12h data directly
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_band = high_roll
    lower_band = low_roll
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if np.isnan(ema_50_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band AND uptrend (price > EMA50) AND volume spike
            if close[i] > upper_band[i] and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band AND downtrend (price < EMA50) AND volume spike
            elif close[i] < lower_band[i] and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below lower Donchian band OR EMA50
            if close[i] < lower_band[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above upper Donchian band OR EMA50
            if close[i] > upper_band[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals