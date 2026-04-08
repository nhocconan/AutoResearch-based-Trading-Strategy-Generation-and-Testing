#!/usr/bin/env python3
# 12h_Donchian_1d_trend_volume_v2
# Hypothesis: Donchian channel breakout on 12h timeframe with 1d trend filter (EMA200) and volume confirmation.
# The Donchian channel (20-period high/low) identifies breakouts with clear support/resistance.
# Trend filter ensures we trade in direction of higher timeframe trend (bullish above EMA200, bearish below).
# Volume confirmation filters out false breakouts. Works in both bull and bear markets by following the trend.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Donchian_1d_trend_volume_v2"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend filter: EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate EMA200 on 1d close
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # 12h Donchian channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after lookback period
    start_idx = lookback
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price breaks below Donchian low OR trend turns bearish
            if close[i] < lowest_low[i] or close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high OR trend turns bullish
            if close[i] > highest_high[i] or close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > 1.5 * vol_avg[i]
            
            # Long: Price breaks above Donchian high + above EMA200 + volume confirmation
            if (close[i] > highest_high[i] and 
                close[i] > ema_200_aligned[i] and 
                vol_confirm):
                position = 1
                signals[i] = 0.25
            # Short: Price breaks below Donchian low + below EMA200 + volume confirmation
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_200_aligned[i] and 
                  vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals