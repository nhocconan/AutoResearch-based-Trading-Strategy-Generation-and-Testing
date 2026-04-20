#!/usr/bin/env python3
# 12h_1d_WickReversal_With_Volume
# Hypothesis: Reversal signals at daily candle wicks on 12h timeframe. Long when 12h closes above prior day's high with volume; short when below prior day's low with volume. Uses 1w EMA50 trend filter to align with higher timeframe bias. Targets 15-30 trades/year per symbol to avoid fee drag, works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_WickReversal_With_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily high and low
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily and weekly data to 12h
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation on 12h
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after weekly EMA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        daily_high_val = daily_high_aligned[i]
        daily_low_val = daily_low_aligned[i]
        ema50_1w_val = ema50_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(daily_high_val) or np.isnan(daily_low_val) or 
            np.isnan(ema50_1w_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above prior day's high with volume, above weekly EMA50
            if (close_val > daily_high_val and vol_ratio_val > 2.0 and 
                close_val > ema50_1w_val):
                signals[i] = 0.25
                position = 1
            # Short: Close below prior day's low with volume, below weekly EMA50
            elif (close_val < daily_low_val and vol_ratio_val > 2.0 and 
                  close_val < ema50_1w_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close back below prior day's high
            if close_val <= daily_high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close back above prior day's low
            if close_val >= daily_low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals