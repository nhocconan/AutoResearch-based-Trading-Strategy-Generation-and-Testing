#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1h timeframe for structure and 1d trend filter.
Trade 1h breakouts of Donchian channels (20-period) with 1d EMA200 trend filter and volume confirmation.
Use 12h only for position sizing and trend filtering to keep trade frequency low (12-30/year).
Works in bull markets via trend-following breakouts and in bear via mean-reversion at 1h structure.
Uses tight entry conditions (3+ confluence) to avoid overtrading.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for structure (Donchian channels)
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    
    # Calculate 1h Donchian channels (20-period)
    high_max_20 = pd.Series(high_1h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1h).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(200) for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1h and 1d data to 12h
    high_max_20_aligned = align_htf_to_ltf(prices, df_1h, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_1h, low_min_20)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume filter: current volume > 1.5x 24-period average (to avoid noise)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC (reduce noise outside active hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available or outside session
        if (np.isnan(high_max_20_aligned[i]) or np.isnan(low_min_20_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 1h Donchian high with volume and above 1d EMA200
            if close[i] > high_max_20_aligned[i] and volume_filter[i] and close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1h Donchian low with volume and below 1d EMA200
            elif close[i] < low_min_20_aligned[i] and volume_filter[i] and close[i] < ema_200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 1h Donchian low (mean reversion)
            if close[i] < low_min_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 1h Donchian high (mean reversion)
            if close[i] > high_max_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1hDonchian20_1dEMA200_Volume_Session"
timeframe = "12h"
leverage = 1.0