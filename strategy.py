#!/usr/bin/env python3
# 4h_1d_donchian_breakout_volume_trend_v1
# Strategy: 4h Donchian breakout with 1d trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Price breaking Donchian(20) channels on 4h captures breakouts, 
# filtered by 1d EMA trend and volume spikes. Works in bull by catching breakouts in uptrend,
# and in bear by catching breakdowns in downtrend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_trend = ema_1d > np.roll(ema_1d, 1)  # Rising EMA = uptrend
    ema_1d_trend[0] = False  # First value has no previous
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_trend)
    
    # Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Breakout conditions
        long_breakout = high[i] > highest_high[i-1]  # Break above prior 20-period high
        short_breakout = low[i] < lowest_low[i-1]    # Break below prior 20-period low
        
        # Exit conditions: opposite breakout or loss of trend
        exit_long = position == 1 and (short_breakout or not ema_1d_aligned[i])
        exit_short = position == -1 and (long_breakout or ema_1d_aligned[i])
        
        # Trading logic: only trade in direction of daily trend
        if long_breakout and ema_1d_aligned[i] and vol_spike[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and not ema_1d_aligned[i] and vol_spike[i] and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals