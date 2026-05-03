#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike
# Donchian breakouts capture momentum moves; 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades.
# Volume spike confirms institutional participation. Designed for low trade frequency (target: 20-50/year) to minimize fee drag.
# Works in both bull and bear markets by trading with the higher timeframe trend.

name = "4h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "4h"
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
    
    # Get 1d data for EMA and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Align 1d indicators to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels for the last 20 periods (4h bars)
        lookback = 20
        start_idx = max(0, i - lookback + 1)
        highest_high = np.max(high[start_idx:i+1])
        lowest_low = np.min(low[start_idx:i+1])
        
        if position == 0:
            # Long: Price breaks above upper Donchian in uptrend with volume spike
            if (high[i] > highest_high and close[i] > ema_50_aligned[i] and volume_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian in downtrend with volume spike
            elif (low[i] < lowest_low and close[i] < ema_50_aligned[i] and volume_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below lower Donchian (reversal)
            if low[i] < lowest_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above upper Donchian (reversal)
            if high[i] > highest_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals