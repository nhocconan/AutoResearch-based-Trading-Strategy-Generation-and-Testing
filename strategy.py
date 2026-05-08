#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian_Breakout_VolumeTrend_v2"
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
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 4-hour Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_channel = high_roll
    lower_channel = low_roll
    
    # Shift to get previous bar's channels (no look-ahead)
    upper_channel_prev = np.roll(upper_channel, 1)
    lower_channel_prev = np.roll(lower_channel, 1)
    upper_channel_prev[0] = np.nan
    lower_channel_prev[0] = np.nan
    
    # Volume spike detection: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(upper_channel_prev[i]) or 
            np.isnan(lower_channel_prev[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper channel with volume spike, above 1d EMA
            if (close[i] > upper_channel_prev[i] and vol_spike and 
                close[i] > ema_val):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel with volume spike, below 1d EMA
            elif (close[i] < lower_channel_prev[i] and vol_spike and 
                  close[i] < ema_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower channel OR below 1d EMA
            if (close[i] < lower_channel_prev[i] or close[i] < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper channel OR above 1d EMA
            if (close[i] > upper_channel_prev[i] or close[i] > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Uses 4-hour Donchian channel breakouts with volume confirmation and 1d EMA trend filter.
# - Enters long when price breaks above upper Donchian channel (previous bar) with volume spike and above 1d EMA
# - Enters short when price breaks below lower Donchian channel (previous bar) with volume spike and below 1d EMA
# - Exits when price breaks back below lower channel (long) or above upper channel (short) OR crosses 1d EMA
# - Volume spike filter ensures breakouts have conviction (threshold lowered to 1.5x to increase signal frequency)
# - 1d EMA filter ensures trading with higher timeframe trend
# - Donchian channels provide clear breakout levels for trend continuation
# - Target: 100-200 total trades over 4 years (25-50/year) to balance frequency and cost
# - Position size: 0.25 for reduced risk exposure
# - Works in both bull and bear markets by following 1d trend direction
# - Volume confirmation reduces false breakouts in low-volume environments
# - Focus on BTC and ETH as primary targets (not SOL-only)