#!/usr/bin/env python3
# 4H_DONCHIAN_BREAKOUT_1D_TREND_VOLUME_CONFIRMATION
# Hypothesis: Donchian(20) breakouts capture momentum. In 1d uptrend (EMA34), go long on upper band breakout with volume > 1.5x 20-period average; in 1d downtrend, go short on lower band breakout with volume confirmation. Works in both bull and bear markets: trend filter avoids counter-trend trades, volume confirms breakout strength.
# Target: 20-50 trades/year on 4h timeframe.

name = "4H_DONCHIAN_BREAKOUT_1D_TREND_VOLUME_CONFIRMATION"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # EMA34 for trend filter
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Donchian channels (20-period) on 4h
    lookback = 20
    highest = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 20)  # Ensure all indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(highest[i]) or np.isnan(lowest[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1d uptrend + price breaks above upper Donchian + volume confirmation
            if (close[i] > ema34_aligned[i] and 
                close[i] > highest[i-1] and 
                volume[i] > 1.5 * vol_avg[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + price breaks below lower Donchian + volume confirmation
            elif (close[i] < ema34_aligned[i] and 
                  close[i] < lowest[i-1] and 
                  volume[i] > 1.5 * vol_avg[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or price breaks below lower Donchian
            if (close[i] <= ema34_aligned[i] or 
                close[i] < lowest[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or price breaks above upper Donchian
            if (close[i] >= ema34_aligned[i] or 
                close[i] > highest[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals