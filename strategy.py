#!/usr/bin/env python3
# 6h_1w_Donchian_Breakout_Volume_TrendFilter
# Hypothesis: Breakouts of weekly Donchian channels (20-week high/low) with volume confirmation and daily trend filter.
# Works in bull via upward breakouts above weekly high, in bear via downward breakouts below weekly low.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Target: 20-40 trades per year per symbol to avoid fee drag, works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Donchian_Breakout_Volume_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Calculate weekly Donchian channels (20-period) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate rolling max/min for Donchian channels
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # === Daily EMA34 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 6h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all weekly and daily levels to 6h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA and volume MA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        ema34_1d_val = ema34_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(donchian_high_val) or np.isnan(donchian_low_val) or 
            np.isnan(ema34_1d_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high with volume confirmation and above daily EMA34
            if (close_val > donchian_high_val and vol_ratio_val > 2.5 and 
                close_val > ema34_1d_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian low with volume confirmation and below daily EMA34
            elif (close_val < donchian_low_val and vol_ratio_val > 2.5 and 
                  close_val < ema34_1d_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below weekly Donchian low
            if close_val <= donchian_low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above weekly Donchian high
            if close_val >= donchian_high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals