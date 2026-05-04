#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Uses Donchian channels for structure on daily timeframe, 1w EMA34 for higher-timeframe trend filter
# (proven effective in capturing major moves while avoiding whipsaw), and volume spike for confirmation.
# Designed for 10-25 trades/year to minimize fee drag. Works in bull markets via breakout continuations
# and in bear markets via breakdown continuations. The 1w EMA34 provides a smooth trend filter that
# adapts to changing regimes while avoiding noise from lower timeframes.

name = "1d_Donchian20_1wEMA34_VolumeSpike_TrendFilter"
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
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 trend filter from prior completed 1w bar
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_shifted = np.roll(ema34_1w, 1)
    ema34_1w_shifted[0] = np.nan
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w_shifted)
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels from prior completed 1d bar
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    lookback = 20
    donchian_upper = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    donchian_upper_shifted = np.roll(donchian_upper, 1)
    donchian_lower_shifted = np.roll(donchian_lower, 1)
    donchian_upper_shifted[0] = np.nan
    donchian_lower_shifted[0] = np.nan
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_shifted)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 1d timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND above 1w EMA34 AND volume spike
            if close[i] > donchian_upper_aligned[i] and close[i] > ema34_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND below 1w EMA34 AND volume spike
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema34_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower OR below 1w EMA34
            if close[i] < donchian_lower_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper OR above 1w EMA34
            if close[i] > donchian_upper_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals