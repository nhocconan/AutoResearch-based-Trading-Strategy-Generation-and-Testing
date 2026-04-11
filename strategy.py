#!/usr/bin/env python3
# 4h_1d_donchian_breakout_volume_v1
# Strategy: 4h Donchian breakout with 1d trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Donchian channel breakouts capture momentum in trending markets.
# Long when price breaks above 20-period high with 1d EMA50 uptrend and volume spike.
# Short when price breaks below 20-period low with 1d EMA50 downtrend and volume spike.
# Volume confirmation reduces false breakouts. Designed for 20-30 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channel (20-period)
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_high.iloc[i]) or 
            np.isnan(lowest_low.iloc[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > highest_high.iloc[i-1]  # Break above previous high
        breakout_down = close[i] < lowest_low.iloc[i-1]  # Break below previous low
        
        # Entry logic: Donchian breakout + volume spike + trend alignment
        if (breakout_up and volume_spike[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (breakout_down and volume_spike[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout or trend change
        elif position == 1 and (breakout_down or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (breakout_up or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals