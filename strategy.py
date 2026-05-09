#!/usr/bin/env python3
# 1d_WeeklyTrend_Breakout_Volume
# Hypothesis: Uses weekly trend direction via 50-period EMA on 1w timeframe to filter entries,
# combined with daily price breaking above/below the previous day's high/low with volume confirmation.
# Trend filter reduces whipsaw in choppy markets, volume confirms breakout strength.
# Designed to work in both bull and bear markets by following the higher timeframe trend.
# Target: 15-25 trades/year per symbol with disciplined risk management.

name = "1d_WeeklyTrend_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (50-period EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly 50-period EMA
    ema_50 = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50[49] = np.mean(close_1w[0:50])  # SMA seed
        alpha = 2.0 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema_50[i] = alpha * close_1w[i] + (1 - alpha) * ema_50[i-1]
    
    # Align weekly EMA to daily timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Previous day's high and low for breakout levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan  # First day has no previous
    prev_low[0] = np.nan
    
    # Volume filter: current volume vs 20-day average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)  # Need volume MA and previous day data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below weekly EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Enter long: uptrend + break above previous day's high + volume confirmation
            if uptrend and close[i] > prev_high[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: downtrend + break below previous day's low + volume confirmation
            elif downtrend and close[i] < prev_low[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal or break below previous day's low
            if not uptrend or close[i] < prev_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal or break above previous day's high
            if not downtrend or close[i] > prev_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals