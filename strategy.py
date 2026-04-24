#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Donchian channel breakouts capture strong momentum moves in both bull and bear markets.
- 12h EMA50 ensures we trade only in the direction of the intermediate trend, reducing whipsaws.
- Volume confirmation (>1.5x 20-bar average) filters low-conviction breakouts.
- Position size 0.25 balances profit potential and drawdown control.
- Target trades: 80-160 total over 4 years (20-40/year) to minimize fee drag.
- Works in bull/bear markets via trend filter and breakout logic.
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
    
    # Get 12h data ONCE before loop for EMA filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian(20) channels from prior 4h bar
    # Upper = max(high[-20:]), Lower = min(low[-20:])
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20) + 1  # Need enough for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms
            if volume_confirm:
                # Long breakout: price above upper band AND above 12h EMA50
                if close[i] > upper[i] and close[i] > ema_50_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price below lower band AND below 12h EMA50
                elif close[i] < lower[i] and close[i] < ema_50_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below lower band OR crosses below 12h EMA50
            if close[i] < lower[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper band OR crosses above 12h EMA50
            if close[i] > upper[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0