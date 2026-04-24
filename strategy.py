#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume spike confirmation.
- Uses Donchian channel (20-day high/low) from 1d timeframe as primary trend structure.
- Breakout above 20-day high with volume > 2.0x 20-bar average = long signal.
- Breakdown below 20-day low with volume > 2.0x 20-bar average = short signal.
- Trend filter: price must be above/below 1w EMA34 to align with weekly trend.
- Designed for 1d timeframe to capture medium-term swings with higher probability entries.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 15-30 trades/year (60-120 total over 4 years) to stay fee-efficient.
- Volume confirmation reduces false breakouts in choppy markets.
- Works in both bull and bear markets by following the weekly trend direction.
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
    
    # Get 1d data ONCE before loop for Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channel (20-period high/low) for 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian high: 20-period rolling high
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian low: 20-period rolling low
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (wait for 1d bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA34 trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough for EMA and Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms breakout
            if volume_confirm:
                # Long: price breaks above Donchian high AND above 1w EMA34
                if close[i] > donchian_high_aligned[i] and close[i] > ema_34_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low AND below 1w EMA34
                elif close[i] < donchian_low_aligned[i] and close[i] < ema_34_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian low OR below 1w EMA34
            if close[i] < donchian_low_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian high OR above 1w EMA34
            if close[i] > donchian_high_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0