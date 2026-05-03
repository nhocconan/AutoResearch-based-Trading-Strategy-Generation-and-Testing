#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and chop regime filter
# Donchian channels provide clear structure for breakouts; volume confirms conviction;
# chop regime (61.8 threshold) avoids whipsaws in ranging markets. Designed for 4h timeframe
# to achieve 20-50 trades/year with discrete position sizing (0.25) to minimize fee drag.
# Works in both bull and bear markets by filtering breakouts with regime and volume.

name = "4h_Donchian20_1dVolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Calculate 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h chop regime (Choppiness Index)
    atr_14 = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / np.log10(14) / (hh_14 - ll_14))
    chop[~np.isfinite(chop)] = 50  # handle division by zero or invalid values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike in low chop (trending)
            if close[i] > high_20[i] and close[i-1] <= high_20[i-1] and volume_spike_aligned[i] and chop[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume spike in low chop (trending)
            elif close[i] < low_20[i] and close[i-1] >= low_20[i-1] and volume_spike_aligned[i] and chop[i] < 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters below Donchian high or high chop (ranging)
            if close[i] < high_20[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters above Donchian low or high chop (ranging)
            if close[i] > low_20[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals