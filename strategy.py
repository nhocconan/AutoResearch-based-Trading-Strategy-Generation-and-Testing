#!/usr/bin/env python3
name = "12h_WeeklyDonchian_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Donchian channels (20 periods)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian upper/lower (20-period high/low)
    donch_upper_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_lower_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA(50) for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 12h
    donch_upper_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_upper_1w)
    donch_lower_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_lower_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Daily volume average (20-period)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align volume to 12h
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_upper_1w_aligned[i]) or np.isnan(donch_lower_1w_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper + price above weekly EMA50 + volume spike
            if (close[i] > donch_upper_1w_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > (1.5 * vol_avg_1d_aligned[i])):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower + price below weekly EMA50 + volume spike
            elif (close[i] < donch_lower_1w_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > (1.5 * vol_avg_1d_aligned[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly Donchian lower
            if close[i] < donch_lower_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above weekly Donchian upper
            if close[i] > donch_upper_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals