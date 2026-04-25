#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 12h EMA50 Trend + Volume Spike
Hypothesis: Donchian breakouts capture strong momentum moves. The 12h EMA50 filter ensures trades align with the medium-term trend, reducing false breakouts. Volume spike confirms breakout strength. Works in bull markets via long breakouts and bear markets via short breakouts. Designed for 4h timeframe to balance trade frequency and signal quality.
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian(20) warmup and EMA alignment
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema_50_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian(20) channels: 20-period high/low
        lookback_start = max(0, i - 19)
        highest_high = np.max(high[lookback_start:i+1])
        lowest_low = np.min(low[lookback_start:i+1])
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        if position == 0:
            # Long: price breaks above 20-period high AND above 12h EMA50 (uptrend filter)
            long_condition = (curr_close > highest_high) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below 20-period low AND below 12h EMA50 (downtrend filter)
            short_condition = (curr_close < lowest_low) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below 20-period low or trend breaks
            if curr_close < lowest_low or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above 20-period high or trend breaks
            if curr_close > highest_high or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0