#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w volume spike + choppiness regime filter
# Donchian breakout captures sustained momentum, 1w volume spike confirms institutional interest,
# choppiness regime ensures we trade in clear trends (CHOP < 38.2) or mean-revert in ranges (CHOP > 61.8).
# Designed to work in both bull and bear markets by adapting to regime.
# Target: 7-25 trades/year (30-100 over 4 years) on 1d timeframe.

name = "1d_Donchian20_1wVolumeSpike_ChopRegime"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for volume spike and choppiness
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1w['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1w['volume'].values > (2.0 * vol_ema_20)
    
    # Calculate 1w choppiness index (CHOP)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) - sum of TR over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(atr_14 / (hh_14 - ll_14)) / log10(14)
    range_14 = hh_14 - ll_14
    chop = 100 * np.log10(atr_14 / range_14) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((range_14 == 0) | np.isnan(chop), 50.0, chop)
    
    # Align 1w indicators to 1d timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Calculate 1d Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: CHOP < 38.2 = trending (breakout), CHOP > 61.8 = ranging (mean revert)
        is_trending = chop_aligned[i] < 38.2
        is_ranging = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: Donchian breakout above upper band + volume spike + (trending OR ranging)
            if high[i] > highest_high[i] and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below lower band + volume spike + (trending OR ranging)
            elif low[i] < lowest_low[i] and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Donchian breakdown below middle band OR reverse signal
            middle = (highest_high[i] + lowest_low[i]) / 2
            if low[i] < middle or (low[i] < lowest_low[i] and volume_spike_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Donchian breakout above middle band OR reverse signal
            middle = (highest_high[i] + lowest_low[i]) / 2
            if high[i] > middle or (high[i] > highest_high[i] and volume_spike_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals