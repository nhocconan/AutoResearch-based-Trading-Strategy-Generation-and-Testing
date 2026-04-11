#!/usr/bin/env python3
# 4h_1d_donchian_volume_chop_v1
# Strategy: 4h Donchian(20) breakout with volume confirmation and chop regime filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture momentum in trending markets, while chop filter avoids false signals in ranging conditions.
# Volume confirmation ensures breakouts are supported by participation. Designed to work in both bull (long breakouts) and bear (short breakouts).
# Uses 1-day Chop Index (14) to filter regimes: < 38.2 = trend (follow breakout), > 61.8 = range (avoid breakouts).
# Target: 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # 1d Chop Index (14) - measures trend vs ranging
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14)
    atr_period = 14
    atr = np.zeros(len(close_1d))
    atr[:atr_period] = np.nan
    if len(close_1d) > atr_period:
        atr[atr_period] = np.nanmean(tr[1:atr_period+1])
        for i in range(atr_period+1, len(close_1d)):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Chop Index = 100 * log10(sum(TR14)/(ATR14 * n)) / log10(n)
    chop = np.full(len(close_1d), np.nan)
    for i in range(atr_period, len(close_1d)):
        if not np.isnan(atr[i]) and atr[i] > 0:
            sum_tr = np.nansum(tr[i-atr_period+1:i+1])
            chop[i] = 100 * np.log10(sum_tr / (atr[i] * atr_period)) / np.log10(atr_period)
    
    # Chop regime: < 38.2 = trend, > 61.8 = range
    chop_regime = chop  # will align and use thresholds
    
    # Align Chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    # Donchian(20) on 4h
    lookback = 20
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest[i] = np.max(high[i-lookback+1:i+1])
        lowest[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(chop_aligned[i]) or np.isnan(highest[i]) or np.isnan(lowest[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Chop regime filter: only trade in trending markets (Chop < 38.2)
        is_trending = chop_aligned[i] < 38.2
        
        # Breakout conditions
        long_breakout = close[i] > highest[i-1]  # break above prior 20-period high
        short_breakout = close[i] < lowest[i-1]  # break below prior 20-period low
        
        # Entry logic: breakout + volume + trend regime
        if long_breakout and vol_confirm[i] and is_trending and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and vol_confirm[i] and is_trending and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout or chop regime shifts to range
        elif position == 1 and (short_breakout or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (long_breakout or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals