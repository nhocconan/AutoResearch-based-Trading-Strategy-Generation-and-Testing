#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with weekly trend filter and volume confirmation.
# Long when price breaks above upper Donchian(20) AND weekly EMA50 rising AND volume > 1.5x 20-period average.
# Short when price breaks below lower Donchian(20) AND weekly EMA50 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside Donchian channel (crosses middle).
# This strategy targets volatility expansion phases with trend alignment to capture momentum moves
# while avoiding choppy markets. The weekly EMA50 filter ensures we trade with the higher timeframe trend.
# Volume confirmation ensures institutional participation and reduces false breakouts.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the weekly trend direction.

name = "12h_DonchianBreakout_WeeklyEMA50_Volume"
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
    
    # Donchian Channel (20)
    dc_length = 20
    upper_dc = pd.Series(high).rolling(window=dc_length, min_periods=dc_length).max().values
    lower_dc = pd.Series(low).rolling(window=dc_length, min_periods=dc_length).min().values
    middle_dc = (upper_dc + lower_dc) / 2.0
    
    # Weekly EMA50 for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Weekly EMA50 direction
    ema50_rising = np.zeros_like(ema50_weekly_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_weekly_aligned, dtype=bool)
    ema50_rising[1:] = ema50_weekly_aligned[1:] > ema50_weekly_aligned[:-1]
    ema50_falling[1:] = ema50_weekly_aligned[1:] < ema50_weekly_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(dc_length, 50)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or np.isnan(middle_dc[i]) or 
            np.isnan(ema50_weekly_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian, weekly EMA50 rising, volume filter
            long_cond = (close[i] > upper_dc[i]) and ema50_rising[i] and volume_filter[i]
            # Short conditions: price breaks below lower Donchian, weekly EMA50 falling, volume filter
            short_cond = (close[i] < lower_dc[i]) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back inside Donchian channel (below middle)
            if close[i] < middle_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back inside Donchian channel (above middle)
            if close[i] > middle_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals