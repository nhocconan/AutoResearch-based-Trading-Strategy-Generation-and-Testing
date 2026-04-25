#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
Long when price breaks above upper Donchian(20) with 1d EMA50 uptrend and volume > 2.0x 20-period average.
Short when price breaks below lower Donchian(20) with 1d EMA50 downtrend and volume > 2.0x 20-period average.
Exit on opposite Donchian touch or trend reversal.
Uses discrete sizing (0.30) to control risk and minimize fee churn.
Target: 20-50 trades/year on 4h timeframe.
Works in bull via trend-following breakouts, in bear via reduced false breakouts due to trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) on primary timeframe (4h)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian and EMA
    start_idx = max(lookback, 50) + 20  # extra buffer for safety
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian with uptrend and volume spike
            long_signal = (close[i] > highest_high[i]) and (close[i] > ema_50_1d_aligned[i]) and vol_spike[i]
            # Short: price breaks below lower Donchian with downtrend and volume spike
            short_signal = (close[i] < lowest_low[i]) and (close[i] < ema_50_1d_aligned[i]) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit conditions: price touches lower Donchian or trend reverses
            exit_signal = (close[i] < lowest_low[i]) or (close[i] < ema_50_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit conditions: price touches upper Donchian or trend reverses
            exit_signal = (close[i] > highest_high[i]) or (close[i] > ema_50_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0