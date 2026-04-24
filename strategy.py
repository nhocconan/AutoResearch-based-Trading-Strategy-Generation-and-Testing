#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
- Donchian(20) breakouts capture strong momentum moves in both bull and bear markets.
- 1d EMA50 ensures we trade only in the direction of the daily trend, reducing whipsaws.
- Volume confirmation (>2.0x 20-bar average) filters low-conviction breakouts.
- Position size 0.25 balances profit potential and drawdown control.
- Target trades: 50-150 total over 4 years (12-37/year) to minimize fee drag.
- Uses 12h primary timeframe with 1d HTF as specified in experiment #81752.
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
    
    # Get 1d data ONCE before loop for EMA filter and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) channels from prior 1d bar
    donchian_high = df_1d['high'].rolling(window=20, min_periods=20).max().values
    donchian_low = df_1d['low'].rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms
            if volume_confirm:
                # Long breakout: price above Donchian high AND above 1d EMA50
                if close[i] > donchian_high_aligned[i] and close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price below Donchian low AND below 1d EMA50
                elif close[i] < donchian_low_aligned[i] and close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR crosses below 1d EMA50
            if close[i] < donchian_low_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR crosses above 1d EMA50
            if close[i] > donchian_high_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0