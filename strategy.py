#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation
# Uses Donchian channel (20-period) breakout for entry, 12h EMA50 for trend filter,
# and volume > 1.5x average for confirmation. Designed to capture strong trends
# in both bull and bear markets while avoiding false breakouts in sideways markets.
# Target: 20-50 trades/year.

name = "4h_Donchian20_12hEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema50_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema50_12h[i] = (close_12h[i] * 2 + ema50_12h[i-1] * 48) / 50
    
    # Align 12h EMA50 to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if required data is NaN
        if np.isnan(ema50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channel (20-period)
        highest_high = np.max(high[i-19:i+1])
        lowest_low = np.min(low[i-19:i+1])
        
        # Volume confirmation: current volume > 1.5x average of last 20 periods
        vol_avg = np.mean(volume[i-20:i]) if i >= 20 else np.mean(volume[:i+1])
        vol_confirm = volume[i] > 1.5 * vol_avg
        
        if position == 0:
            # Look for entry: Donchian breakout with trend filter and volume confirmation
            long_breakout = close[i] > highest_high
            short_breakout = close[i] < lowest_low
            
            # Trend filter: price above/below 12h EMA50
            uptrend = close[i] > ema50_12h_aligned[i]
            downtrend = close[i] < ema50_12h_aligned[i]
            
            if long_breakout and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            elif short_breakout and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian middle or trend reverses
            donchian_middle = (highest_high + lowest_low) / 2
            if close[i] < donchian_middle or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian middle or trend reverses
            donchian_middle = (highest_high + lowest_low) / 2
            if close[i] > donchian_middle or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals