#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w/1d pivot direction filter and volume confirmation.
# Long: price breaks above Donchian(20) high AND price > 1w pivot point (bullish bias) AND volume > 1.5x 20-bar average.
# Short: price breaks below Donchian(20) low AND price < 1w pivot point (bearish bias) AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed to capture breakouts in both bull and bear markets.

name = "6h_Donchian20_1wPivot_Direction_VolumeConfirm_v1"
timeframe = "6h"
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
    
    # Load 1w data ONCE before loop for pivot point calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot point: (weekly_high + weekly_low + weekly_close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    pivot_1w = (weekly_high + weekly_low + weekly_close) / 3.0
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Load 1d data ONCE before loop for additional trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA34 and pivot
    
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
        if np.isnan(pivot_1w_aligned[i]) or np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_pivot_1w = pivot_1w_aligned[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        # Donchian(20) calculation requires 20 periods of high/low
        if i < 20 + start_idx:  # need extra warmup for Donchian
            signals[i] = 0.0
            continue
            
        # Calculate Donchian(20) channels
        highest_high = np.max(high[i-19:i+1])  # 20 periods including current
        lowest_low = np.min(low[i-19:i+1])
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        curr_vol_ma = vol_ma[i]
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high AND price > 1w pivot AND price > 1d EMA34 AND volume confirmation
            if (curr_close > highest_high and 
                curr_close > curr_pivot_1w and 
                curr_close > curr_ema_34_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND price < 1w pivot AND price < 1d EMA34 AND volume confirmation
            elif (curr_close < lowest_low and 
                  curr_close < curr_pivot_1w and 
                  curr_close < curr_ema_34_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low OR price < 1w pivot (bias change)
            if (curr_close < lowest_low or 
                curr_close < curr_pivot_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR price > 1w pivot (bias change)
            if (curr_close > highest_high or 
                curr_close > curr_pivot_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals