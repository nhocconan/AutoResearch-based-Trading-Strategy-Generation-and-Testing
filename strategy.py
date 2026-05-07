#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w EMA20 trend filter and volume confirmation.
# Long when price breaks above upper Donchian(20) AND 1w EMA20 rising AND volume > 1.5x 20-period average.
# Short when price breaks below lower Donchian(20) AND 1w EMA20 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside Donchian channel (crosses middle line).
# This strategy targets volatility expansion phases with trend alignment to capture momentum moves
# while avoiding choppy markets. The 1w EMA20 filter ensures we trade with the higher timeframe trend.
# Volume confirmation ensures institutional participation and reduces false breakouts.
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 1w trend direction.

name = "1d_DonchianBreakout_WeeklyEMA20_Volume"
timeframe = "1d"
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
    upper_channel = pd.Series(high).rolling(window=dc_length, min_periods=dc_length).max().values
    lower_channel = pd.Series(low).rolling(window=dc_length, min_periods=dc_length).min().values
    middle_channel = (upper_channel + lower_channel) / 2
    
    # 1w EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # 1w EMA20 direction
    ema20_rising = np.zeros_like(ema20_1w_aligned, dtype=bool)
    ema20_falling = np.zeros_like(ema20_1w_aligned, dtype=bool)
    ema20_rising[1:] = ema20_1w_aligned[1:] > ema20_1w_aligned[:-1]
    ema20_falling[1:] = ema20_1w_aligned[1:] < ema20_1w_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(dc_length, 20)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(middle_channel[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(ema20_rising[i]) or np.isnan(ema20_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian, 1w EMA20 rising, volume filter
            long_cond = (close[i] > upper_channel[i]) and ema20_rising[i] and volume_filter[i]
            # Short conditions: price breaks below lower Donchian, 1w EMA20 falling, volume filter
            short_cond = (close[i] < lower_channel[i]) and ema20_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back inside Donchian channel (below middle line)
            if close[i] < middle_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back inside Donchian channel (above middle line)
            if close[i] > middle_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals