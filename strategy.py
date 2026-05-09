#!/usr/bin/env python3
# 4H_Donchian20_Breakout_1dTrend_Volume
# Hypothesis: On 4h timeframe, enter long when price breaks above 20-period Donchian high with 1d uptrend and volume confirmation.
# Short when price breaks below 20-period Donchian low with 1d downtrend and volume confirmation.
# Uses 1d EMA(34) trend filter to avoid counter-trend trades and Donchian channels for breakout signals.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years).

name = "4H_Donchian20_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d trend: EMA(34) on close
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = close_1d > ema_34
    
    # Donchian channel (20-period) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Align 1d trend to 4h
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(trend_up_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_confirm[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high + 1d uptrend + volume confirmation
            if close[i] > donchian_high[i] and trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + 1d downtrend + volume confirmation
            elif close[i] < donchian_low[i] and not trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend changes
            if close[i] < donchian_low[i] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend changes
            if close[i] > donchian_high[i] or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals