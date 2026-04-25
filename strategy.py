#!/usr/bin/env python3
"""
1h Donchian Breakout + 4h EMA Trend + Volume Spike + Session Filter (08-20 UTC)
Hypothesis: Donchian channel breakouts capture trending moves; 4h EMA filters trend direction (long above EMA, short below); volume spike confirms breakout strength; session filter (08-20 UTC) reduces noise during low-liquidity hours. Works in bull/bear by trend-filtering breakouts. Target: 15-37 trades/year (60-150 over 4 years).
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
    open_time = prices['open_time'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Donchian channels (20-period) on 1h
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate 20-period average volume for volume spike
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian (20) + volume MA (20) + EMA50 warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma_20[i]) or not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        donchian_upper = highest_high[i]
        donchian_lower = lowest_low[i]
        vol_ma = vol_ma_20[i]
        volume_spike = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: break above Donchian upper + above 4h EMA50 + volume spike
            long_condition = (curr_close > donchian_upper) and (curr_close > ema_trend) and volume_spike
            # Short: break below Donchian lower + below 4h EMA50 + volume spike
            short_condition = (curr_close < donchian_lower) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint or trend breaks
            midpoint = (donchian_upper + donchian_lower) / 2.0
            if curr_close <= midpoint or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to Donchian midpoint or trend breaks
            midpoint = (donchian_upper + donchian_lower) / 2.0
            if curr_close >= midpoint or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian_Breakout_4hEMA50_Trend_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0