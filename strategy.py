#!/usr/bin/env python3
name = "1d_WeeklyDonchianBreakout_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channels (20-week high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels
    high_20w = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Weekly EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.5 x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure Donchian and volume MA data
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Break above weekly Donchian high in uptrend with volume confirmation
            if close[i] > donchian_high_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly Donchian low in downtrend with volume confirmation
            elif close[i] < donchian_low_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to weekly Donchian midpoint
            donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2.0
            if position == 1 and close[i] < donchian_mid:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals