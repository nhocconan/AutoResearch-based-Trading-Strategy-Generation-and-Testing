#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation
# Uses Donchian channels for structure, 12h EMA50 for trend filter, and volume spike for confirmation.
# Designed for 12-25 trades/year to minimize fee drag. Works in bull markets via upside breakouts
# and in bear markets via downside breakdowns. Novel for 6h timeframe with 12h HTF filter.

name = "6h_Donchian20_12hEMA50_VolumeSpike_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels and EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA50 trend filter from prior completed 12h bar
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_shifted = np.roll(ema50_12h, 1)
    ema50_12h_shifted[0] = np.nan
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h_shifted)
    
    # Calculate Donchian channels (20-period) from prior completed 12h bar
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift to use prior completed 12h bar levels
    donchian_upper_shifted = np.roll(donchian_upper, 1)
    donchian_lower_shifted = np.roll(donchian_lower, 1)
    donchian_upper_shifted[0] = np.nan
    donchian_lower_shifted[0] = np.nan
    
    # Align Donchian channels to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_shifted)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian upper AND 12h EMA50 uptrend AND volume spike
            if close[i] > donchian_upper_aligned[i] and close[i] > ema50_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian lower AND 12h EMA50 downtrend AND volume spike
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema50_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower OR below 12h EMA50
            if close[i] < donchian_lower_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper OR above 12h EMA50
            if close[i] > donchian_upper_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals