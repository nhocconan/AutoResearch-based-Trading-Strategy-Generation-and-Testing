#!/usr/bin/env python3
"""
12h_VolumeSpike_RangeBreakout_1dTrend
Hypothesis: Breakouts of daily range (high-low) on 12h timeframe with volume spike (>2x average) and aligned with 1-day EMA50 trend. Works in bull markets by buying strength and in bear markets by selling weakness. Uses 1d for trend and range, 12h for precise entry/exit to limit trades to 15-30/year.
"""

name = "12h_VolumeSpike_RangeBreakout_1dTrend"
timeframe = "12h"
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
    
    # Get 1d data for range and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate daily range for breakout levels
    daily_range = daily_high - daily_low
    # Breakout levels: above daily high or below daily low
    breakout_high = daily_high  # Break above prior day high
    breakout_low = daily_low    # Break below prior day low
    
    # Align daily levels to 12h timeframe (with 1-bar delay for completed 1d bar)
    breakout_high_aligned = align_htf_to_ltf(prices, df_1d, breakout_high, additional_delay_bars=1)
    breakout_low_aligned = align_htf_to_ltf(prices, df_1d, breakout_low, additional_delay_bars=1)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    
    # 1d trend filter: EMA50
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average on 12h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(breakout_high_aligned[i]) or 
            np.isnan(breakout_low_aligned[i]) or 
            np.isnan(daily_close_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = daily_close_aligned[i] > ema_50_1d_aligned[i]
        trend_down = daily_close_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above prior day high with upward trend and volume spike
            if (close[i] > breakout_high_aligned[i] and 
                trend_up and 
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below prior day low with downward trend and volume spike
            elif (close[i] < breakout_low_aligned[i] and 
                  trend_down and 
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to prior day close or trend turns down
            if close[i] < daily_close_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to prior day close or trend turns up
            if close[i] > daily_close_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals