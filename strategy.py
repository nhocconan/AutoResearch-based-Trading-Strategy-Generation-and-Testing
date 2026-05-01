#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + volume confirmation + 1w trend filter.
# Uses 1w EMA50 for major trend direction to avoid counter-trend trades.
# Long when: price breaks above Donchian(20) high AND volume > 1.5x volume MA(20) AND price > 1w EMA50.
# Short when: price breaks below Donchian(20) low AND volume > 1.5x volume MA(20) AND price < 1w EMA50.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 12-25 trades/year.
# Donchian channels provide structural breakouts; volume confirms conviction; 1w EMA50 filters regime.

name = "12h_Donchian20_VolumeConfirm_1wTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian(20) channels
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: volume > 1.5x volume MA(20)
    volume_ma = np.full(n, np.nan)
    for i in range(19, n):
        volume_ma[i] = np.mean(volume[i-19:i+1])
    volume_ratio = np.full(n, np.nan)
    for i in range(19, n):
        if volume_ma[i] > 0:
            volume_ratio[i] = volume[i] / volume_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_ratio[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        curr_volume_ratio = volume_ratio[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        
        # Breakout conditions
        breakout_up = curr_close > curr_highest_high
        breakout_down = curr_close < curr_lowest_low
        volume_confirmed = curr_volume_ratio > 1.5
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish breakout + volume confirmation + above 1w EMA50
            if (breakout_up and 
                volume_confirmed and 
                curr_close > curr_ema_50_1w):
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + volume confirmation + below 1w EMA50
            elif (breakout_down and 
                  volume_confirmed and 
                  curr_close < curr_ema_50_1w):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low OR loses volume confirmation
            if (curr_close < curr_lowest_low or 
                curr_volume_ratio < 1.2):  # slightly looser exit for volume
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR loses volume confirmation
            if (curr_close > curr_highest_high or 
                curr_volume_ratio < 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals