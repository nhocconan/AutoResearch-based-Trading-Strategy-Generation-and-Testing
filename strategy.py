#!/usr/bin/env python3
# 1d_WeeklyDonchian_TrailingExit
# Hypothesis: On daily timeframe, enter long when price breaks above weekly Donchian high (20-period),
# enter short when price breaks below weekly Donchian low. Exit when price crosses the weekly midpoint.
# Use volume confirmation (2.0x average) to avoid false breakouts. Weekly trend filter: only long when
# price above weekly 50-period SMA, only short when price below weekly 50-period SMA.
# This captures medium-term trends with low trade frequency, suitable for 1d timeframe.
# Weekly data acts as trend filter and breakout trigger, reducing whipsaw in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyDonchian_TrailingExit"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Weekly Donchian channels (20-period) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian high and low
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    # Weekly midpoint for exit
    weekly_midpoint = (donchian_high + donchian_low) / 2.0
    # Weekly 50-period SMA for trend filter
    sma_50 = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Align to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    weekly_midpoint_aligned = align_htf_to_ltf(prices, df_1w, weekly_midpoint)
    sma_50_aligned = align_htf_to_ltf(prices, df_1w, sma_50)
    
    # === Daily volume confirmation ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for weekly indicators
        # Get values
        close_val = prices['close'].iloc[i]
        high_val = donchian_high_aligned[i]
        low_val = donchian_low_aligned[i]
        midpoint_val = weekly_midpoint_aligned[i]
        sma_val = sma_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(high_val) or np.isnan(low_val) or np.isnan(midpoint_val) or 
            np.isnan(sma_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high with volume confirmation AND above weekly SMA (uptrend)
            if close_val > high_val and vol_ratio_val > 2.0 and close_val > sma_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian low with volume confirmation AND below weekly SMA (downtrend)
            elif close_val < low_val and vol_ratio_val > 2.0 and close_val < sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below weekly midpoint
            if close_val < midpoint_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above weekly midpoint
            if close_val > midpoint_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals