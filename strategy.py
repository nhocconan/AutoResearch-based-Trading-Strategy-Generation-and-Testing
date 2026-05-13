#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation (>2.0x 20-bar avg).
# Uses 1d EMA34 for trend alignment, 6h Donchian channels for breakout entry, and volume spike for confirmation.
# Designed for low trade frequency (target 50-150 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 1d trend direction and requiring strong volume confirmation to avoid false breakouts.

name = "6h_Donchian20_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels from prior 6h bar (primary TF)
    lookback = 20
    # Upper channel = highest high of last 20 periods, lower channel = lowest low of last 20 periods
    # Using prior bar's data to avoid look-ahead
    upper_channel = pd.Series(high).shift(1).rolling(window=lookback, min_periods=lookback).max().values
    lower_channel = pd.Series(low).shift(1).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate average volume for confirmation (20-period LTF)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian channel, close > 1d EMA34, volume spike (>2.0x avg)
            if (high[i] > upper_channel[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian channel, close < 1d EMA34, volume spike (>2.0x avg)
            elif (low[i] < lower_channel[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price breaks below lower Donchian channel or volume drops significantly
            if (low[i] < lower_channel[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if price breaks above upper Donchian channel or volume drops significantly
            if (high[i] > upper_channel[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals