#!/usr/bin/env python3
# 12h_1d_Donchian_Breakout_Volume_TrendFilter
# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter.
# Long when price breaks above upper band with volume > 2x average and price > EMA50.
# Short when price breaks below lower band with volume > 2x average and price < EMA50.
# Uses breakouts as clean momentum signals in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Donchian_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate 12h Donchian channels (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper and lower bands (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Daily EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align daily EMA50 to 12h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Get values
        close_val = close[i]
        upper_band = high_roll[i]
        lower_band = low_roll[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_band) or np.isnan(lower_band) or np.isnan(ema50_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper band with volume confirmation and above daily EMA50
            if (close_val > upper_band and  # Break above Donchian upper
                vol_ratio_val > 2.0 and     # Volume confirmation
                close_val > ema50_val):     # Above daily EMA50 (uptrend filter)
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band with volume confirmation and below daily EMA50
            elif (close_val < lower_band and   # Break below Donchian lower
                  vol_ratio_val > 2.0 and      # Volume confirmation
                  close_val < ema50_val):      # Below daily EMA50 (downtrend filter)
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to middle of channel or breaks lower band
            mid_band = (upper_band + lower_band) / 2.0
            if close_val < mid_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to middle of channel or breaks upper band
            mid_band = (upper_band + lower_band) / 2.0
            if close_val > mid_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals