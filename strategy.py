#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrendFilter_VolumeSpike_v1
Hypothesis: Trade 12h Donchian(20) breakouts with 1w trend filter and volume confirmation.
Long when price breaks above Donchian(20) high and 1w close > 1w EMA50 (bull trend).
Short when price breaks below Donchian(20) low and 1w close < 1w EMA50 (bear trend).
Volume confirmation: volume > 1.5 * ATR(14) to avoid false breakouts.
ATR-based stoploss: exit when price moves against position by 2.5 * ATR(14).
Only trade in direction of 1w trend to avoid counter-trend whipsaws.
Target: 12-30 trades/year to minimize fee drag while capturing sustained moves.
Discrete sizing: 0.25.
Works in both bull and bear markets by only trading with the 1w trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for 12h data (for volume spike and stoploss)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first TR undefined
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian(20) channels for 12h data
    # Donchian high = max(high, lookback=20)
    # Donchian low = min(low, lookback=20)
    # Use pandas rolling for efficiency
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track holding period
    
    # Start index: need warmup for Donchian(20) (20), ATR(14) (14), and 1w EMA50 (50)
    start_idx = max(20, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume spike: current volume > 1.5 * ATR (adaptive threshold)
        volume_spike = volume[i] > 1.5 * atr[i]
        
        # Determine 1w trend regime
        # Bull trend: close > EMA50
        # Bear trend: close < EMA50
        if close[i] > ema_50_1w_aligned[i]:
            regime = 'bull'  # only allow longs
        elif close[i] < ema_50_1w_aligned[i]:
            regime = 'bear'  # only allow shorts
        else:
            regime = 'neutral'  # no trades (rare)
        
        if position == 0:
            # Long setup: price breaks above Donchian high AND volume spike AND bull regime
            long_setup = (close[i] > donchian_high[i]) and volume_spike and (regime == 'bull')
            
            # Short setup: price breaks below Donchian low AND volume spike AND bear regime
            short_setup = (close[i] < donchian_low[i]) and volume_spike and (regime == 'bear')
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            elif short_setup:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            bars_since_entry += 1
            # Exit: price closes below Donchian low (breakdown) OR ATR stoploss (2.5 * ATR)
            if (close[i] < donchian_low[i]) or (close[i] <= high_since_entry - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit: price closes above Donchian high (breakout) OR ATR stoploss (2.5 * ATR)
            if (close[i] > donchian_high[i]) or (close[i] >= low_since_entry + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        
        # Track highest high and lowest low since entry for ATR stoploss
        if position == 1:
            if bars_since_entry == 0:
                high_since_entry = high[i]
            else:
                high_since_entry = max(high_since_entry, high[i])
        elif position == -1:
            if bars_since_entry == 0:
                low_since_entry = low[i]
            else:
                low_since_entry = min(low_since_entry, low[i])
    
    return signals

name = "12h_Donchian20_Breakout_1wTrendFilter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0