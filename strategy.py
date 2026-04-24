#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA34 trend filter and volume spike confirmation.
- Uses Donchian channel (20-period high/low) from 12h timeframe as price structure.
- Breakout above upper band with volume > 2.0x 20-bar average = long signal.
- Breakdown below lower band with volume > 2.0x 20-bar average = short signal.
- Trend filter: price must be above/below 1w EMA34 to align with weekly trend.
- Designed for 12h timeframe to capture multi-day swings with higher probability entries.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-37 trades/year (50-150 total over 4 years) to stay fee-efficient.
- Volume confirmation reduces false breakouts in choppy markets.
- Novelty: Uses weekly EMA34 on 12h timeframe for stronger trend alignment than daily.
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
    
    # Get 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA34 trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Donchian channel (20-period) on 12h timeframe
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough for weekly EMA and Donchian/volume
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms breakout
            if volume_confirm:
                # Long: price breaks above upper Donchian AND above 1w EMA34
                if close[i] > donchian_upper[i] and close[i] > ema_34_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below lower Donchian AND below 1w EMA34
                elif close[i] < donchian_lower[i] and close[i] < ema_34_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price crosses below lower Donchian OR below 1w EMA34
            if close[i] < donchian_lower[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above upper Donchian OR above 1w EMA34
            if close[i] > donchian_upper[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1wEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0