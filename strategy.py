#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses Donchian channels from 1d timeframe for structure, 1w EMA50 for trend filter,
# and volume spike for confirmation. Designed for 15-25 trades/year to minimize fee drag.
# Works in bull markets via upside breakouts and in bear markets via downside breakdowns.
# The Donchian channels provide adaptive support/resistance that captures volatility expansion.

name = "1d_Donchian20_1wEMA50_VolumeSpike_TrendFilter"
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
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 trend filter from prior completed 1w bar
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_shifted = np.roll(ema50_1w, 1)
    ema50_1w_shifted[0] = np.nan
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w_shifted)
    
    # Calculate Donchian(20) channels from prior completed 1d bar
    # Upper channel = max(high, lookback=20), Lower channel = min(low, lookback=20)
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift to use prior completed 1d bar levels
    high_max_20_shifted = np.roll(high_max_20, 1)
    low_min_20_shifted = np.roll(low_min_20, 1)
    high_max_20_shifted[0] = np.nan
    low_min_20_shifted[0] = np.nan
    
    # Align Donchian channels to 1d timeframe (no shift needed as already 1d)
    upper_channel = high_max_20_shifted
    lower_channel = low_min_20_shifted
    
    # Volume confirmation: 20-period EMA of volume on 1d timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above upper Donchian channel AND 1w EMA50 uptrend AND volume spike
            if close[i] > upper_channel[i] and close[i] > ema50_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower Donchian channel AND 1w EMA50 downtrend AND volume spike
            elif close[i] < lower_channel[i] and close[i] < ema50_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below lower Donchian channel OR below 1w EMA50
            if close[i] < lower_channel[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above upper Donchian channel OR above 1w EMA50
            if close[i] > upper_channel[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals