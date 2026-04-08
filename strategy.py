#!/usr/bin/env python3
# 1d_donchian_breakout_1w_trend_volume_v1
# Hypothesis: Daily Donchian breakout (20-day) in direction of weekly trend (price above/below weekly EMA50).
# Volume filter (current volume > 1.5x 20-day average) ensures participation.
# Works in bull markets by capturing breakouts and in bear markets by capturing breakdowns.
# Target: 15-30 trades/year with ~0.25 position size to minimize fee drag.

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average (20-day)
    vol_ma = np.zeros_like(volume)
    vol_ma[19:] = np.convolve(volume, np.ones(20)/20, mode='valid')
    vol_ma[:19] = vol_ma[19]
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Weekly EMA (50-period) for trend filter
    ema_period = 50
    ema_weekly = np.zeros_like(close_weekly)
    if len(close_weekly) >= ema_period:
        ema_weekly[ema_period-1] = np.mean(close_weekly[:ema_period])
        for i in range(ema_period, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * 2 + ema_weekly[i-1] * (ema_period - 1)) / (ema_period + 1)
    
    # Align weekly EMA to daily timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Donchian channels (20-day)
    lookback = 20
    upper = np.full_like(high, np.nan)
    lower = np.full_like(low, np.nan)
    for i in range(lookback-1, n):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(lookback, ema_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vol_ma[i]) or volume[i] == 0 or 
            np.isnan(ema_weekly_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Higher timeframe trend filter: price above/below weekly EMA
        uptrend_htf = close[i] > ema_weekly_aligned[i]
        downtrend_htf = close[i] < ema_weekly_aligned[i]
        
        if position == 1:  # Long position
            # Exit if trend reverses, volume fails, or price breaks below lower Donchian
            if not uptrend_htf or not volume_filter or close[i] < lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if trend reverses, volume fails, or price breaks above upper Donchian
            if not downtrend_htf or not volume_filter or close[i] > upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian, volume breakout, and weekly uptrend
            if close[i] > upper[i] and volume_filter and uptrend_htf:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian, volume breakout, and weekly downtrend
            elif close[i] < lower[i] and volume_filter and downtrend_htf:
                position = -1
                signals[i] = -0.25
    
    return signals