#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
- Uses Donchian channel (20-period high/low) from 4h timeframe as price structure.
- Breakout above 20-period high with volume > 2.0x 20-bar average = long signal.
- Breakdown below 20-period low with volume > 2.0x 20-bar average = short signal.
- Trend filter: price must be above/below 1d EMA34 to align with daily trend.
- Designed for 4h timeframe to capture swing trades with higher probability entries.
- Uses discrete position size 0.30 to limit drawdown and reduce fee churn.
- Targets 20-50 trades/year (75-200 total over 4 years) to stay fee-efficient.
- Volume confirmation reduces false breakouts in choppy markets.
- Novelty: Combines Donchian breakout with 1d EMA34 on 4h timeframe for BTC/ETH edge.
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
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian(20) on 4h timeframe
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms breakout
            if volume_confirm:
                # Long: price breaks above 20-period high AND above 1d EMA34
                if close[i] > high_roll[i] and close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.30
                    position = 1
                # Short: price breaks below 20-period low AND below 1d EMA34
                elif close[i] < low_roll[i] and close[i] < ema_34_1d_aligned[i]:
                    signals[i] = -0.30
                    position = -1
        elif position == 1:
            # Long exit: price crosses below 20-period low OR below 1d EMA34
            if close[i] < low_roll[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price crosses above 20-period high OR above 1d EMA34
            if close[i] > high_roll[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0