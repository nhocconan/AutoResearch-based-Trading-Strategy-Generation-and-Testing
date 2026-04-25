#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + 1w EMA50 Trend + Volume Spike
Hypothesis: Donchian channel breakouts capture strong momentum moves. The 1w EMA50 filters for primary trend direction, ensuring we only trade breakouts in the direction of the weekly trend. Volume spike confirms genuine interest. Works in both bull and bear markets via trend filter - in bear markets, we only take short breaks below Donchian lower band when weekly trend is down, and vice versa. Target 20-50 trades/year to minimize fee drag.
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian(20) and EMA50 warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if EMA50 not ready
        if np.isnan(ema_50_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        
        # Donchian(20): 20-period high/low
        lookback = min(20, i+1)
        period_high = np.max(high[i-lookback+1:i+1])
        period_low = np.min(low[i-lookback+1:i+1])
        
        # Volume spike: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 1.5 * vol_ma_20
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND above 1w EMA50 (uptrend filter)
            long_condition = (curr_close > period_high) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below Donchian lower band AND below 1w EMA50 (downtrend filter)
            short_condition = (curr_close < period_low) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian middle or trend breaks
            donchian_mid = (period_high + period_low) / 2.0
            if curr_close < donchian_mid or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian middle or trend breaks
            donchian_mid = (period_high + period_low) / 2.0
            if curr_close > donchian_mid or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0