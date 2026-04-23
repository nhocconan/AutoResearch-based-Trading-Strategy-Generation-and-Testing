#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian channel breakout with 1d EMA200 trend filter and ATR-based volume spike confirmation.
Donchian(20) captures significant price momentum, EMA200 ensures alignment with long-term trend,
and volume spikes confirm institutional participation. Works in both bull and bear markets via trend filter.
Target: 12-37 trades/year per symbol (50-150 total over 4 years) with discrete sizing (0.25) to minimize fees.
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
    
    # Calculate Donchian channel (20-period) on 6h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate ATR(14) for volatility and volume spike detection
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume MA (20-period) for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 200, 20)  # need Donchian, EMA200, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA200 = uptrend, close < 1d EMA200 = downtrend
        trend_up = close[i] > ema_200_1d_aligned[i]
        trend_down = close[i] < ema_200_1d_aligned[i]
        
        # Volume filter: 6h volume > 2.0x 20-period MA (volume spike)
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_20[i-1]  # break above previous period's high
        breakout_down = close[i] < lowest_20[i-1]  # break below previous period's low
        
        if position == 0:
            # Long: Donchian breakout up AND uptrend AND volume spike
            if breakout_up and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND downtrend AND volume spike
            elif breakout_down and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Donchian breakout in opposite direction
            exit_signal = False
            if position == 1:
                # Exit long on Donchian breakdown
                if breakout_down:
                    exit_signal = True
            elif position == -1:
                # Exit short on Donchian breakout up
                if breakout_up:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_Breakout_1dEMA200_VolumeSpike"
timeframe = "6h"
leverage = 1.0