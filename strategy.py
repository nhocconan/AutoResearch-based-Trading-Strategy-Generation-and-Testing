#!/usr/bin/env python3
name = "1D_WeeklyDonchian_Breakout_VolumeTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Donchian channel and trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian channel (20-week)
    donchian_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA(34) for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly indicators to daily
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike: current volume > 1.8x 50-day average
    vol_avg = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_spike = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + above weekly EMA34 + volume spike
            if (close[i] > donchian_high_20_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low + below weekly EMA34 + volume spike
            elif (close[i] < donchian_low_20_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly Donchian low or below weekly EMA34
            if close[i] < donchian_low_20_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above weekly Donchian high or above weekly EMA34
            if close[i] > donchian_high_20_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals