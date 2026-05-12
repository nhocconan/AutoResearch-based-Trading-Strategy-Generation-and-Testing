#!/usr/bin/env python3
# 1D_WEEKLY_BREAKOUT_CONFIRMED
# Hypothesis: Break above/below weekly Donchian(20) channel with volume confirmation and price above/below weekly EMA34.
# Works in both bull and bear markets: weekly EMA34 filter ensures we trade with higher timeframe trend,
# while Donchian breakout captures momentum. Volume confirmation reduces false breakouts.
# Target: 10-20 trades/year on 1d timeframe.

name = "1D_WEEKLY_BREAKOUT_CONFIRMED"
timeframe = "1d"
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
    
    # Weekly data for Donchian channels and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    # Weekly EMA34 for trend filter
    ema34 = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly indicators to daily timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34)
    
    # Volume confirmation: current volume > 1.5x weekly average volume
    vol_avg = pd.Series(df_1w['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1w, vol_avg)
    volume_confirmed = volume > (1.5 * vol_avg_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly Donchian high + above weekly EMA34 + volume confirmation
            if (close[i] > high_20_aligned[i] and 
                close[i] > ema34_aligned[i] and 
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Donchian low + below weekly EMA34 + volume confirmation
            elif (close[i] < low_20_aligned[i] and 
                  close[i] < ema34_aligned[i] and 
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly Donchian low or closes below weekly EMA34
            if (close[i] < low_20_aligned[i] or 
                close[i] < ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly Donchian high or closes above weekly EMA34
            if (close[i] > high_20_aligned[i] or 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals