#!/usr/bin/env python3
# 12h_1d_Donchian20_Breakout_Volume_TrendFilter
# Hypothesis: Breakouts of 12h Donchian(20) channel with volume confirmation and 1d EMA50 trend filter.
# Uses daily EMA50 to filter trend direction (only trade long when above, short when below).
# Target: 15-35 trades per year per symbol to avoid fee flood, works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Donchian20_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for Donchian and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # === Calculate 12h Donchian(20) channel ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Donchian upper (20-period high) and lower (20-period low)
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # === 12h EMA21 for trend filter (additional confirmation) ===
    close_12h_series = pd.Series(close_12h)
    ema21_12h = close_12h_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # === Daily data for EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 12h and daily levels to 12h
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA and volume MA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        ema21_12h_val = ema21_12h_aligned[i]
        ema50_1d_val = ema50_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(donch_high_val) or np.isnan(donch_low_val) or np.isnan(ema21_12h_val) or 
            np.isnan(ema50_1d_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 12h Donchian high with volume confirmation, above 12h EMA21, and above daily EMA50
            if (close_val > donch_high_val and vol_ratio_val > 2.0 and 
                close_val > ema21_12h_val and close_val > ema50_1d_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 12h Donchian low with volume confirmation, below 12h EMA21, and below daily EMA50
            elif (close_val < donch_low_val and vol_ratio_val > 2.0 and 
                  close_val < ema21_12h_val and close_val < ema50_1d_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below 12h Donchian low
            if close_val <= donch_low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above 12h Donchian high
            if close_val >= donch_high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals