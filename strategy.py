#!/usr/bin/env python3
# 1d_Weekly_Donchian_Breakout_Volume_TrendFilter
# Hypothesis: Daily breakouts of weekly Donchian channels with volume confirmation and weekly EMA trend filter.
# Long when price breaks above weekly upper band with volume surge and above weekly EMA.
# Short when price breaks below weekly lower band with volume surge and below weekly EMA.
# Exit when price returns to weekly midline or opposite band.
# Designed for low trade frequency (~15-25/year) to minimize fee drag and work in bull/bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Donchian_Breakout_Volume_TrendFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for Donchian and EMA
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # === Calculate weekly Donchian channels (20-period) ===
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Donchian upper and lower bands
    upper_weekly = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    lower_weekly = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    mid_weekly = (upper_weekly + lower_weekly) / 2.0
    
    # === Weekly EMA34 for trend filter ===
    close_weekly_series = pd.Series(close_weekly)
    ema34_weekly = close_weekly_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all weekly levels to daily
    upper_weekly_aligned = align_htf_to_ltf(prices, df_weekly, upper_weekly)
    lower_weekly_aligned = align_htf_to_ltf(prices, df_weekly, lower_weekly)
    mid_weekly_aligned = align_htf_to_ltf(prices, df_weekly, mid_weekly)
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # === Daily: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA and volume MA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        upper_val = upper_weekly_aligned[i]
        lower_val = lower_weekly_aligned[i]
        mid_val = mid_weekly_aligned[i]
        ema34_val = ema34_weekly_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_val) or np.isnan(lower_val) or np.isnan(mid_val) or 
            np.isnan(ema34_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly upper band with volume confirmation and above weekly EMA
            if (close_val > upper_val and vol_ratio_val > 2.0 and close_val > ema34_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly lower band with volume confirmation and below weekly EMA
            elif (close_val < lower_val and vol_ratio_val > 2.0 and close_val < ema34_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to weekly midline or below lower band
            if close_val <= mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to weekly midline or above upper band
            if close_val >= mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals