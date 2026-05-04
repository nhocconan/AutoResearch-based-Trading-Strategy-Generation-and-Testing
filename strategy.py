#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation
# Uses Donchian channels from 12h chart for price structure, 1d EMA50 for trend filter, and volume spike for confirmation.
# Designed for 12-37 trades/year to minimize fee drag. Works in bull markets via breakout continuations and in bear markets via breakdown continuations.
# The 1d EMA50 provides a smooth trend filter that adapts to changing regimes while avoiding whipsaw.

name = "12h_Donchian20_1dEMA50_VolumeSpike_TrendFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter from prior completed 1d bar
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_shifted = np.roll(ema50_1d, 1)
    ema50_1d_shifted[0] = np.nan
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_shifted)
    
    # Calculate 12h Donchian(20) from prior completed 12h bar
    donchian_period = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = low_series.rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_high_shifted = np.roll(donchian_high, 1)
    donchian_high_shifted[0] = np.nan
    donchian_low_shifted = np.roll(donchian_low, 1)
    donchian_low_shifted[0] = np.nan
    donchian_high_aligned = align_htf_to_ltf(prices, high_series.index.to_frame(index=False), donchian_high_shifted)
    donchian_low_aligned = align_htf_to_ltf(prices, low_series.index.to_frame(index=False), donchian_low_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND above 1d EMA50 AND volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > ema50_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND below 1d EMA50 AND volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < ema50_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian low OR below 1d EMA50
            if close[i] < donchian_low_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian high OR above 1d EMA50
            if close[i] > donchian_high_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals