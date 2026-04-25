#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 12h EMA50 Trend + Volume Spike
Hypothesis: Donchian breakouts capture strong momentum, filtered by 12h EMA50 trend direction to avoid counter-trend trades. Volume spike confirms institutional participation. Works in bull/bear markets by trend-filtering breakouts. Target: 20-50 trades/year (80-200 over 4 years).
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian(20) + EMA50 warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout signals with trend filter
        if position == 0:
            # Long: price breaks above upper Donchian channel AND above 12h EMA50 (uptrend)
            long_condition = (curr_close > upper_channel) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below lower Donchian channel AND below 12h EMA50 (downtrend)
            short_condition = (curr_close < lower_channel) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.30
                position = 1
            elif short_condition:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of channel or trend breaks
            midpoint = (upper_channel + lower_channel) / 2.0
            if curr_close <= midpoint or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price returns to midpoint of channel or trend breaks
            midpoint = (upper_channel + lower_channel) / 2.0
            if curr_close >= midpoint or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0