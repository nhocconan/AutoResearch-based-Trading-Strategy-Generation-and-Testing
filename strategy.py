# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_donchian_breakout_volume_trend_v1
# Breakouts of 4h Donchian(20) with volume confirmation (>1.5x 20-period average) and 1d trend filter (EMA50).
# Works in both bull and bear markets by capturing strong momentum moves aligned with daily trend.
# Low trade frequency expected (~20-40/year) due to strict breakout conditions + volume + trend filter.
name = "4h_1d_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate volume average (20-period)
    vol_avg = np.full(n, np.nan)
    for i in range(lookback-1, n):
        vol_avg[i] = np.mean(volume[i-lookback+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if indicators not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        bullish_breakout = close[i] > highest_high[i-1]  # break above previous high
        bearish_breakout = close[i] < lowest_low[i-1]    # break below previous low
        
        # Volume confirmation (>1.5x average volume)
        volume_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Trend filter: align with 1d EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Entry signals
        long_entry = bullish_breakout and volume_confirm and uptrend
        short_entry = bearish_breakout and volume_confirm and downtrend
        
        # Exit conditions: opposite breakout or trend change
        exit_long = bearish_breakout and volume_confirm
        exit_short = bullish_breakout and volume_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals