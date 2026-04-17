#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d/2d structure and 1w trend filter.
Trade 1d breakouts of Donchian channels with 1w EMA200 trend filter and volume confirmation.
Use 12h only for precise entry timing to keep trade frequency low (15-30/year).
Works in bull markets via trend-following breakouts and in bear via mean-reversion at 1d structure.
"""
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
    
    # Get 1d data for structure (Donchian channels)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA(200) for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d and 1w data to 12h
    high_max_20_aligned = align_htf_to_ltf(prices, df_1d, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_1d, low_min_20)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume filter: current volume > 1.5x 12-period average (to avoid noise)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 00-23 UTC (full day coverage for 12h timeframe)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 0) & (hours <= 23)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available or outside session
        if (np.isnan(high_max_20_aligned[i]) or np.isnan(low_min_20_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian high with volume and above 1w EMA200
            if close[i] > high_max_20_aligned[i] and volume_filter[i] and close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian low with volume and below 1w EMA200
            elif close[i] < low_min_20_aligned[i] and volume_filter[i] and close[i] < ema_200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 1d Donchian low (mean reversion)
            if close[i] < low_min_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 1d Donchian high (mean reversion)
            if close[i] > high_max_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dDonchian20_1wEMA200_Volume"
timeframe = "12h"
leverage = 1.0