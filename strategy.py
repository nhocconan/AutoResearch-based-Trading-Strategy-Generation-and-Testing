#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above upper Donchian(20) AND 1w EMA50 rising AND volume > 1.5x 20-period average.
# Short when price breaks below lower Donchian(20) AND 1w EMA50 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside Donchian channels.
# This strategy targets volatility expansion phases with trend alignment to capture momentum moves
# while avoiding choppy markets. The 1w EMA50 filter ensures we trade with the higher timeframe trend.
# Volume confirmation ensures institutional participation and reduces false breakouts.
# Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 1w trend direction.

name = "1d_DonchianBreakout_1wEMA50_Volume"
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
    
    # Donchian Channels (20)
    dc_length = 20
    upper_dc = pd.Series(high).rolling(window=dc_length, min_periods=dc_length).max().values
    lower_dc = pd.Series(low).rolling(window=dc_length, min_periods=dc_length).min().values
    middle_dc = (upper_dc + lower_dc) / 2.0
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1w EMA50 direction
    ema50_rising = np.zeros_like(ema50_1w_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_1w_aligned, dtype=bool)
    ema50_rising[1:] = ema50_1w_aligned[1:] > ema50_1w_aligned[:-1]
    ema50_falling[1:] = ema50_1w_aligned[1:] < ema50_1w_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(dc_length, 50)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or np.isnan(middle_dc[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema50_rising[i]) or np.isnan(ema50_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian, 1w EMA50 rising, volume filter
            long_cond = (close[i] > upper_dc[i]) and ema50_rising[i] and volume_filter[i]
            # Short conditions: price breaks below lower Donchian, 1w EMA50 falling, volume filter
            short_cond = (close[i] < lower_dc[i]) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back inside Donchian channels (below middle)
            if close[i] < middle_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back inside Donchian channels (above middle)
            if close[i] > middle_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals