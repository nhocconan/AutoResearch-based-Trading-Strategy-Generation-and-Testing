#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h 1w Donchian Breakout with Weekly Trend Filter and Volume Confirmation
# Hypothesis: Use weekly Donchian channels to capture major trends, filtered by weekly price position
# relative to 200-period EMA, with volume confirmation to avoid false breakouts. Works in bull
# (breakouts above upper band) and bear (breakdowns below lower band) markets by following the
# weekly trend. Targets 15-30 trades/year to minimize fee drag.
name = "12h_1w_Donchian20_WeeklyTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly Indicators ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian Channel (20-period)
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA200 for trend filter
    close_series = pd.Series(close_1w)
    ema200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly indicators to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    ema200_aligned = align_htf_to_ltf(prices, df_1w, ema200)
    
    # === 12h Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Get values
        close_val = prices['close'].iloc[i]
        high_20_val = high_20_aligned[i]
        low_20_val = low_20_aligned[i]
        ema200_val = ema200_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(close_val) or np.isnan(high_20_val) or 
            np.isnan(low_20_val) or np.isnan(ema200_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above weekly Donchian high with volume, price above weekly EMA200
            if close_val > high_20_val and vol_ratio_val > 2.0 and close_val > ema200_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly Donchian low with volume, price below weekly EMA200
            elif close_val < low_20_val and vol_ratio_val > 2.0 and close_val < ema200_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below weekly Donchian low
            if close_val < low_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above weekly Donchian high
            if close_val > high_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals