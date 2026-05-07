#!/usr/bin/env python3
name = "1d_WeeklyDonchian_Trend_4hVOL"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (EMA10)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly EMA10 for trend filter
    close_1w = df_1w['close'].values
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Get weekly data for Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper: highest high of last 20 weeks
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian lower: lowest low of last 20 weeks
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # Get 4h data for volume filter (volume spike)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    vol_4h = df_4h['volume'].values
    vol_avg = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol_4h > (vol_avg * 1.5)  # 50% above average
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_10_1w_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above weekly EMA10 (uptrend), daily close above weekly Donchian high, 4h volume spike
            if (close[i] > ema_10_1w_aligned[i] and 
                close[i] > donch_high_aligned[i] and 
                volume_spike_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: price below weekly EMA10 (downtrend), daily close below weekly Donchian low, 4h volume spike
            elif (close[i] < ema_10_1w_aligned[i] and 
                  close[i] < donch_low_aligned[i] and 
                  volume_spike_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly EMA10 (trend change) or touches weekly Donchian low
            if close[i] < ema_10_1w_aligned[i] or close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above weekly EMA10 (trend change) or touches weekly Donchian high
            if close[i] > ema_10_1w_aligned[i] or close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals