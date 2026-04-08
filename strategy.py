#!/usr/bin/env python3
# 12h_donchian_breakout_1d_trend_volume_v1
# Hypothesis: 12h Donchian breakout (price breaks 20-period high/low) in direction of daily trend (price above/below daily EMA50) with volume confirmation (volume > 1.5x 20-period average).
# Works in bull markets by capturing continuation breakouts and in bear markets by capturing breakdowns.
# Volume filter ensures participation, trend filter avoids counter-trend trades.
# Target: 15-30 trades/year with ~0.25 position size to minimize fee drag.

name = "12h_donchian_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average (20-period)
    vol_ma = np.zeros_like(volume)
    vol_ma[19:] = np.convolve(volume, np.ones(20)/20, mode='valid')
    vol_ma[:19] = vol_ma[19]
    
    # Donchian channels (20-period)
    high_max = np.full_like(high, np.nan)
    low_min = np.full_like(low, np.nan)
    for i in range(19, n):
        high_max[i] = np.max(high[i-19:i+1])
        low_min[i] = np.min(low[i-19:i+1])
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    
    # Daily EMA (50-period) for trend filter
    ema_period = 50
    ema_daily = np.zeros_like(close_daily)
    if len(close_daily) >= ema_period:
        ema_daily[ema_period-1] = np.mean(close_daily[:ema_period])
        for i in range(ema_period, len(close_daily)):
            ema_daily[i] = (close_daily[i] * 2 + ema_daily[i-1] * (ema_period - 1)) / (ema_period + 1)
    
    # Align daily EMA to 12h timeframe
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(20, ema_period) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(vol_ma[i]) or volume[i] == 0 or 
            np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Higher timeframe trend filter: price above/below daily EMA
        uptrend_htf = close[i] > ema_daily_aligned[i]
        downtrend_htf = close[i] < ema_daily_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_max[i-1]  # Break above previous 20-period high
        breakout_down = close[i] < low_min[i-1]  # Break below previous 20-period low
        
        if position == 1:  # Long position
            # Exit if trend reverses, volume fails, or price breaks below Donchian low
            if not uptrend_htf or not volume_filter or close[i] < low_min[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if trend reverses, volume fails, or price breaks above Donchian high
            if not downtrend_htf or not volume_filter or close[i] > high_max[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Donchian breakout up, volume breakout, and daily uptrend
            if breakout_up and volume_filter and uptrend_htf:
                position = 1
                signals[i] = 0.25
            # Short entry: Donchian breakout down, volume breakout, and daily downtrend
            elif breakout_down and volume_filter and downtrend_htf:
                position = -1
                signals[i] = -0.25
    
    return signals