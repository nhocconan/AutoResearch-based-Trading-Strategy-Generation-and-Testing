#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume spike confirmation, and ATR-based trailing stop.
- Donchian(20) provides clear structure for breakouts in both bull and bear markets.
- 1d EMA34 ensures alignment with higher timeframe trend to reduce counter-trend trades.
- Volume > 2.0x 20-bar average confirms breakout strength and reduces false signals.
- ATR trailing stop (3.0x ATR) manages risk and adapts to volatility.
- Discrete position size 0.25 limits drawdown and reduces fee churn.
- Target: 20-50 trades/year on 4h timeframe to stay fee-efficient.
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
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channels (20-period) for breakout levels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility-based trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(34, lookback, 20, 14)  # EMA34, Donchian20, VolMA20, ATR14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only enter on volume-confirmed breakouts
            if volume_confirm:
                # Long: price breaks above Donchian high AND above 1d EMA34
                if close[i] > highest_high[i] and close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    highest_since_entry = close[i]
                # Short: price breaks below Donchian low AND below 1d EMA34
                elif close[i] < lowest_low[i] and close[i] < ema_34_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    lowest_since_entry = close[i]
        elif position == 1:
            # Update highest close since entry for trailing stop
            highest_since_entry = max(highest_since_entry, close[i])
            # Long exit: price closes below 1d EMA34 OR trailing stop hit
            trail_stop = highest_since_entry - 3.0 * atr[i]
            if close[i] < ema_34_1d_aligned[i] or close[i] < trail_stop:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest close since entry for trailing stop
            lowest_since_entry = min(lowest_since_entry, close[i])
            # Short exit: price closes above 1d EMA34 OR trailing stop hit
            trail_stop = lowest_since_entry + 3.0 * atr[i]
            if close[i] > ema_34_1d_aligned[i] or close[i] > trail_stop:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0