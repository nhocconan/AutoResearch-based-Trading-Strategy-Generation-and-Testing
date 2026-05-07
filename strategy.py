#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d trend filter and volume confirmation.
# Long when price breaks above upper Donchian(20) AND 1d EMA34 rising AND volume > 1.5x 20-period average.
# Short when price breaks below lower Donchian(20) AND 1d EMA34 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside Donchian channels (crosses middle line).
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 1d trend direction and capturing volatility breakouts.

name = "12h_DonchianBreakout_1dEMA34_Volume"
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
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(dc_length, 34)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or np.isnan(middle_dc[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_rising[i]) or np.isnan(ema34_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian, 1d EMA34 rising, volume filter
            long_cond = (close[i] > upper_dc[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price breaks below lower Donchian, 1d EMA34 falling, volume filter
            short_cond = (close[i] < lower_dc[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back inside Donchian (below middle)
            if close[i] < middle_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back inside Donchian (above middle)
            if close[i] > middle_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals