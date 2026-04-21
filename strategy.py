#!/usr/bin/env python3
"""
4h_1d_Donchian20_Breakout_Volume_TrendFilter_v1
Hypothesis: Donchian(20) breakout on 4h timeframe with 1d EMA50 trend filter and volume confirmation.
Works in bull/bear: In uptrend (price > 1d EMA50), take long breakouts; in downtrend (price < 1d EMA50), take short breakouts.
Volume confirmation avoids false breakouts. Target: 20-50 trades/year per symbol (80-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels on 4h data (primary timeframe)
    lookback = 20
    highest = prices['high'].rolling(window=lookback, min_periods=lookback).max().values
    lowest = prices['low'].rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=1).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if indicators not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest[i]) or np.isnan(lowest[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] if i > 0 else False
        downtrend = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] if i > 0 else False
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND uptrend AND volume
            if price > highest[i] and uptrend and volume_ok:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian lower band AND downtrend AND volume
            elif price < lowest[i] and downtrend and volume_ok:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian lower band or trend reverses
            if price < lowest[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: price breaks above Donchian upper band or trend reverses
            if price > highest[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_1d_Donchian20_Breakout_Volume_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0