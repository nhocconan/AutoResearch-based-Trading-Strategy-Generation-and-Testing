#!/usr/bin/env python3
# 12h_12H_382Rule_Breakout_1dTrend_Volume
# Hypothesis: 38.2% rule breakout from daily range with 1d trend filter and volume confirmation.
# The 38.2% level (Fibonacci retracement) is a key support/resistance level that often acts as
# a pivot point for reversals or continuations. In trending markets, price tends to respect
# this level as support in uptrends and resistance in downtrends. Combines with 1d EMA trend
# filter to avoid counter-trend trades and volume spike to confirm breakout strength.
# Designed for low trade frequency (~15-25/year) to minimize fee drag on 12h timeframe.

name = "12h_12H_382Rule_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Get daily data for 38.2% rule calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for 38.2% rule calculation
    ph = np.concatenate([[high_1d[0]], high_1d[:-1]])  # previous high
    pl = np.concatenate([[low_1d[0]], low_1d[:-1]])   # previous low
    pc = np.concatenate([[close_1d[0]], close_1d[:-1]]) # previous close
    
    # Calculate 38.2% rule levels (key retracement level)
    rang = ph - pl
    # In uptrend: resistance at 38.2% retracement from low to high
    # In downtrend: support at 38.2% retracement from high to low
    res_382 = pl + 0.382 * rang  # 38.2% level from low
    sup_382 = ph - 0.382 * rang  # 38.2% level from high
    
    # Align 38.2% levels to 12h timeframe
    res_382_aligned = align_htf_to_ltf(prices, df_1d, res_382)
    sup_382_aligned = align_htf_to_ltf(prices, df_1d, sup_382)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (ema_34_1d[i-1] * 33 + close_1d[i]) / 34
    
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(res_382_aligned[i]) or np.isnan(sup_382_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 38.2% resistance AND uptrend (price > EMA34) AND volume spike
            if (close[i] > res_382_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 38.2% support AND downtrend (price < EMA34) AND volume spike
            elif (close[i] < sup_382_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 38.2% support OR trend reversal (price < EMA34)
            if close[i] < sup_382_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 38.2% resistance OR trend reversal (price > EMA34)
            if close[i] > res_382_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals