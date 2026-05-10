#!/usr/bin/env python3
# 1d_RangeBreakout_1wTrend_Volume
# Hypothesis: Daily range breakout with weekly trend filter and volume confirmation.
# Enters long when price breaks above previous day's high with volume > 1.5x 20-day average and weekly close > weekly SMA50.
# Enters short when price breaks below previous day's low with volume > 1.5x 20-day average and weekly close < weekly SMA50.
# Exits when price returns to the opposite day's extreme (e.g., long exits when price < previous day's low).
# Uses weekly SMA50 for trend to avoid whipsaws and works in both bull/bear markets.
# Targets 15-25 trades per year on 1d timeframe with position size 0.25.

name = "1d_RangeBreakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly SMA(50) for trend
    sma_50_1w = pd.Series(df_1w['close']).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_ltf_to_htf(prices, df_1w, sma_50_1w)
    
    # Previous day's high and low for breakout levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = high[0]  # first day uses current day's high
    prev_low[0] = low[0]    # first day uses current day's low
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for weekly SMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(sma_50_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: weekly close above/below SMA50
        weekly_close = df_1w['close'].iloc[-1] if len(df_1w) > 0 else 0
        weekly_sma = sma_50_1w_aligned[i]
        weekly_uptrend = weekly_close > weekly_sma
        weekly_downtrend = weekly_close < weekly_sma
        
        if position == 0:
            # Long entry: price breaks above previous day's high with volume confirmation and weekly uptrend
            if (high[i] > prev_high[i] and 
                volume_confirm[i] and 
                weekly_uptrend):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below previous day's low with volume confirmation and weekly downtrend
            elif (low[i] < prev_low[i] and 
                  volume_confirm[i] and 
                  weekly_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to previous day's low
            if low[i] < prev_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to previous day's high
            if high[i] > prev_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals