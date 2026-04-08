#!/usr/bin/env python3
# 12h_donchian_breakout_1d_trend_volume_v1
# Hypothesis: 12h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation.
# Long when price breaks above 20-bar high + price > 1d EMA50 + volume > 1.5x avg volume.
# Short when price breaks below 20-bar low + price < 1d EMA50 + volume > 1.5x avg volume.
# Exit on opposite Donchian breakout or volume failure.
# Designed to capture trend continuations in both bull and bear markets with controlled trade frequency.
# Target: 20-40 trades/year.

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
    
    # Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i - 19:i + 1])
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    
    # Daily EMA (50-period)
    ema_period = 50
    ema_daily = np.full_like(close_daily, np.nan)
    for i in range(ema_period - 1, len(close_daily)):
        if i == ema_period - 1:
            ema_daily[i] = np.mean(close_daily[:ema_period])
        else:
            ema_daily[i] = (close_daily[i] * 2 + ema_daily[i-1] * (ema_period - 1)) / (ema_period + 1)
    
    # Align daily EMA to 12h timeframe
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(lookback, ema_period) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Higher timeframe trend filter
        uptrend_htf = close[i] > ema_daily_aligned[i]
        downtrend_htf = close[i] < ema_daily_aligned[i]
        
        if position == 1:  # Long position
            # Exit on bearish Donchian break or volume failure
            if close[i] < lowest_low[i] or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on bullish Donchian break or volume failure
            if close[i] > highest_high[i] or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: bullish Donchian break, volume confirmation, and daily uptrend
            if (close[i] > highest_high[i] and 
                volume_filter and 
                uptrend_htf):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish Donchian break, volume confirmation, and daily downtrend
            elif (close[i] < lowest_low[i] and 
                  volume_filter and 
                  downtrend_htf):
                position = -1
                signals[i] = -0.25
    
    return signals