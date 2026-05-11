#!/usr/bin/env python3
name = "1d_1w_VolumeWeighted_Breakout"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1-week data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1-week Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donch_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donch_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_high_1w)
    donch_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_low_1w)
    
    # 1-week volume-weighted average price (VWAP)
    typical_price_1w = (high_1w + low_1w + df_1w['close'].values) / 3.0
    vwap_1w = (typical_price_1w * df_1w['volume'].values).cumsum() / df_1w['volume'].values.cumsum()
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # Daily volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        if np.isnan(donch_high_1w_aligned[i]) or np.isnan(donch_low_1w_aligned[i]) or np.isnan(vwap_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above 1w Donchian high AND above 1w VWAP + volume
            if close[i] > donch_high_1w_aligned[i] and close[i] > vwap_1w_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 1w Donchian low AND below 1w VWAP + volume
            elif close[i] < donch_low_1w_aligned[i] and close[i] < vwap_1w_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below 1w Donchian low or below 1w VWAP
            if close[i] < donch_low_1w_aligned[i] or close[i] < vwap_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above 1w Donchian high or above 1w VWAP
            if close[i] > donch_high_1w_aligned[i] or close[i] > vwap_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals