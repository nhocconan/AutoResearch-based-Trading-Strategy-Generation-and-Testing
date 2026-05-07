#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter, volume confirmation, and ATR stoploss.
# Long when price breaks above Donchian(20) upper band AND 12h EMA60 rising AND volume > 1.3x 20-period average.
# Short when price breaks below Donchian(20) lower band AND 12h EMA60 falling AND volume > 1.3x 20-period average.
# Exit when price crosses back inside Donchian channel.
# Uses discrete position sizing (0.25) to minimize churn. Designed for low trade frequency (<40/year) to avoid fee drag.
# Works in both bull and bear markets by following 12h trend direction.

name = "4h_DonchianBreakout_12hEMA60_Volume"
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
    
    # Donchian Channel (20)
    dc_length = 20
    upper_dc = pd.Series(high).rolling(window=dc_length, min_periods=dc_length).max().values
    lower_dc = pd.Series(low).rolling(window=dc_length, min_periods=dc_length).min().values
    
    # 12h EMA60 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema60_12h = pd.Series(close_12h).ewm(span=60, adjust=False, min_periods=60).mean().values
    ema60_12h_aligned = align_htf_to_ltf(prices, df_12h, ema60_12h)
    
    # 12h EMA60 direction
    ema60_rising = np.zeros_like(ema60_12h_aligned, dtype=bool)
    ema60_falling = np.zeros_like(ema60_12h_aligned, dtype=bool)
    ema60_rising[1:] = ema60_12h_aligned[1:] > ema60_12h_aligned[:-1]
    ema60_falling[1:] = ema60_12h_aligned[1:] < ema60_12h_aligned[:-1]
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(dc_length, 60)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or np.isnan(ema60_12h_aligned[i]) or 
            np.isnan(ema60_rising[i]) or np.isnan(ema60_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper DC, 12h EMA60 rising, volume filter
            long_cond = (close[i] > upper_dc[i]) and ema60_rising[i] and volume_filter[i]
            # Short conditions: price breaks below lower DC, 12h EMA60 falling, volume filter
            short_cond = (close[i] < lower_dc[i]) and ema60_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back inside Donchian channel (below lower band)
            if close[i] < lower_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back inside Donchian channel (above upper band)
            if close[i] > upper_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals