#!/usr/bin/env python3
# 12h_ema_crossover_volume_filter_v1
# Hypothesis: 12h EMA crossover (fast/slow) with volume confirmation and daily trend filter.
# Uses fast EMA crossing above/below slow EMA for entry, volume > 1.5x average for confirmation,
# and daily EMA trend filter to avoid counter-trend trades. Works in bull markets by catching
# uptrend continuations and in bear markets by catching downtrend continuations.
# Volume filter reduces false signals, targeting 20-40 trades/year.

name = "12h_ema_crossover_volume_filter_v1"
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
    
    # EMA indicators
    fast_period = 9
    slow_period = 21
    
    # Fast EMA
    ema_fast = np.zeros_like(close)
    ema_fast[fast_period-1] = np.mean(close[:fast_period])
    for i in range(fast_period, len(close)):
        ema_fast[i] = (close[i] * 2 + ema_fast[i-1] * (fast_period - 1)) / (fast_period + 1)
    
    # Slow EMA
    ema_slow = np.zeros_like(close)
    ema_slow[slow_period-1] = np.mean(close[:slow_period])
    for i in range(slow_period, len(close)):
        ema_slow[i] = (close[i] * 2 + ema_slow[i-1] * (slow_period - 1)) / (slow_period + 1)
    
    # EMA crossover signals
    ema_crossover = np.where(ema_fast > ema_slow, 1, -1)  # 1 for bullish, -1 for bearish
    
    # Volume filter: 20-period average volume
    vol_ma = np.zeros_like(volume)
    vol_ma[19:] = np.convolve(volume, np.ones(20)/20, mode='valid')
    vol_ma[:19] = vol_ma[19]  # Fill beginning with first valid value
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    
    # Daily EMA (50-period) for higher timeframe trend
    ema_period = 50
    ema_daily = np.zeros_like(close_daily)
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
    start_idx = max(fast_period, slow_period, ema_period) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Higher timeframe trend filter: price above/below daily EMA
        uptrend_htf = close[i] > ema_daily_aligned[i]
        downtrend_htf = close[i] < ema_daily_aligned[i]
        
        if position == 1:  # Long position
            # Exit if trend reverses or volume fails
            if ema_crossover[i] == -1 or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if trend reverses or volume fails
            if ema_crossover[i] == 1 or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: bullish EMA crossover, volume confirmation, and daily uptrend
            if (ema_crossover[i] == 1 and 
                volume_filter and 
                uptrend_htf):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish EMA crossover, volume confirmation, and daily downtrend
            elif (ema_crossover[i] == -1 and 
                  volume_filter and 
                  downtrend_htf):
                position = -1
                signals[i] = -0.25
    
    return signals