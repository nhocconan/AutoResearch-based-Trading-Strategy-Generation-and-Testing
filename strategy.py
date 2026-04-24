#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 12h EMA200 trend filter and volume spike confirmation.
- Uses Donchian(20) channels from 1d timeframe as dynamic support/resistance.
- Breakout above upper channel with volume > 2.0x 20-bar average = long signal.
- Breakdown below lower channel with volume > 2.0x 20-bar average = short signal.
- Trend filter: price must be above/below 12h EMA200 to align with higher timeframe direction.
- Designed for 4h timeframe to capture medium-term trends with proper channel structure.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 19-50 trades/year (75-200 total over 4 years) to stay fee-efficient.
- Combines proven Donchian breakout with 12h trend filter (better than 1d for 4h timeframe).
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
    
    # Get 1d data ONCE before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Donchian channels for 1d timeframe (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper = 20-period high, lower = 20-period low
    upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h timeframe (wait for 1d bar to close)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # Get 12h data ONCE before loop for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA200 trend filter
    close_12h = df_12h['close'].values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(200, 20)  # Need enough for EMA200 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_200_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms breakout
            if volume_confirm:
                # Long: price breaks above upper channel AND above 12h EMA200
                if close[i] > upper_aligned[i] and close[i] > ema_200_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below lower channel AND below 12h EMA200
                elif close[i] < lower_aligned[i] and close[i] < ema_200_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price crosses below lower channel OR below 12h EMA200
            if close[i] < lower_aligned[i] or close[i] < ema_200_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above upper channel OR above 12h EMA200
            if close[i] > upper_aligned[i] or close[i] > ema_200_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA200_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0