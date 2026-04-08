#!/usr/bin/env python3
# 4h_volume_breakout_1d_trend_v1
# Hypothesis: 4h volume breakout (volume > 2x 20-period average) in direction of daily trend (price above/below daily EMA50).
# Works in bull markets by capturing continuation breakouts and in bear markets by capturing breakdowns.
# Volume filter ensures participation, trend filter avoids counter-trend trades.
# Target: 20-40 trades/year with ~0.25 position size to minimize fee drag.

name = "4h_volume_breakout_1d_trend_v1"
timeframe = "4h"
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
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    
    # Daily EMA (50-period) for trend filter
    ema_period = 50
    ema_daily = np.zeros_like(close_daily)
    ema_daily[ema_period-1] = np.mean(close_daily[:ema_period])
    for i in range(ema_period, len(close_daily)):
        ema_daily[i] = (close_daily[i] * 2 + ema_daily[i-1] * (ema_period - 1)) / (ema_period + 1)
    
    # Align daily EMA to 4h timeframe
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
            np.isnan(ema_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period average
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        # Higher timeframe trend filter: price above/below daily EMA
        uptrend_htf = close[i] > ema_daily_aligned[i]
        downtrend_htf = close[i] < ema_daily_aligned[i]
        
        if position == 1:  # Long position
            # Exit if trend reverses or volume fails
            if not uptrend_htf or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if trend reverses or volume fails
            if not downtrend_htf or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: volume breakout and daily uptrend
            if volume_filter and uptrend_htf:
                position = 1
                signals[i] = 0.25
            # Short entry: volume breakout and daily downtrend
            elif volume_filter and downtrend_htf:
                position = -1
                signals[i] = -0.25
    
    return signals