#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrendFilter_VolumeSpike
Hypothesis: Trade 6h Donchian(20) breakouts with 1d EMA34 trend filter and volume confirmation.
Uses ATR-based volume spike filter (volume > 1.5 * ATR) and discrete sizing (0.25) to reduce trades.
Designed for 12-37 trades/year to minimize fee drag while maintaining edge in both bull and bear markets.
ATR volatility filter adapts to changing market conditions better than fixed MA volume average.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for volume spike filter (adaptive to volatility)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first TR undefined
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian(20) channels on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d EMA34 (34) and ATR (14) and Donchian (20)
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 1.5 * ATR (adaptive threshold)
        volume_spike = volume[i] > 1.5 * atr[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND 1d trend bullish (close > EMA34) AND volume spike
            long_setup = (close[i] > highest_high[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike
            # Short: price breaks below lower Donchian AND 1d trend bearish (close < EMA34) AND volume spike
            short_setup = (close[i] < lowest_low[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
                          volume_spike
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters Donchian channel OR 1d trend turns bearish
            if (lowest_low[i] <= close[i] <= highest_high[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Donchian channel OR 1d trend turns bullish
            if (lowest_low[i] <= close[i] <= highest_high[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1dTrendFilter_VolumeSpike"
timeframe = "6h"
leverage = 1.0