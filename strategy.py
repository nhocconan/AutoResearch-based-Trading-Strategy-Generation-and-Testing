#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w high/low filter and volume confirmation.
# Uses weekly Donchian channels as trend filter: only take long breaks above 6h Donchian high
# when price is above 1w Donchian mean, and short breaks below 6h Donchian low when price
# is below 1w Donchian mean. Volume confirmation ensures breakout strength.
# Works in bull/bear by following higher timeframe trend and avoiding counter-trend breakouts.
# Target: 15-35 trades per year to minimize fee drag.

name = "6h_Donchian20_1wTrendFilter_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1w Donchian mean for trend filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mean_1w = (donchian_high_1w + donchian_low_1w) / 2.0
    donchian_mean_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_mean_1w)
    
    # === 6h Donchian(20) breakout levels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        dh_6h = donchian_high[i]
        dl_6h = donchian_low[i]
        trend_filter = donchian_mean_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(dh_6h) or np.isnan(dl_6h) or np.isnan(trend_filter) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 6h Donchian high, above 1w Donchian mean, with volume
            if high_val > dh_6h and close_val > trend_filter and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6h Donchian low, below 1w Donchian mean, with volume
            elif low_val < dl_6h and close_val < trend_filter and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below 6h Donchian low or trend reverses
            if close_val < dl_6h or close_val < trend_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above 6h Donchian high or trend reverses
            if close_val > dh_6h or close_val > trend_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals