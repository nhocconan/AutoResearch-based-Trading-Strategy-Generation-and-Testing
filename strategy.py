#!/usr/bin/env python3
# 1d_1w_Donchian_Breakout_Volume_TrendFilter
# Hypothesis: 1d Donchian breakout with volume confirmation and 1w EMA trend filter.
# Long when price breaks above 1d Donchian upper band with volume spike and above 1w EMA.
# Short when price breaks below 1d Donchian lower band with volume spike and below 1w EMA.
# Exit when price returns to Donchian middle band. Designed for fewer trades (<30/year) to avoid fee drag.
# Works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Donchian_Breakout_Volume_TrendFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === 1w EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 1d Donchian channels (20-period) ===
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Donchian upper and lower bands
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # === 1d: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align 1w EMA to daily
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA and Donchian warmup
        # Get values
        close_val = close_1d[i]
        upper_val = donchian_upper[i]
        lower_val = donchian_lower[i]
        middle_val = donchian_middle[i]
        ema34_1w_val = ema34_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_val) or np.isnan(lower_val) or np.isnan(middle_val) or 
            np.isnan(ema34_1w_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper with volume confirmation and above 1w EMA
            if (close_val > upper_val and vol_ratio_val > 2.0 and close_val > ema34_1w_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with volume confirmation and below 1w EMA
            elif (close_val < lower_val and vol_ratio_val > 2.0 and close_val < ema34_1w_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below Donchian middle
            if close_val <= middle_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above Donchian middle
            if close_val >= middle_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals