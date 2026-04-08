#!/usr/bin/env python3
# 4h_ema_pullback_volume_surge_v1
# Hypothesis: In trending markets, price pulls back to the 21-period EMA on 4h charts with high volume (>2.5x average) present high-probability continuation entries.
# Uses EMA trend filter (price > EMA50 for longs, < EMA50 for shorts) to ensure alignment with higher timeframe trend.
# Volume surge confirms institutional interest during pullback. Works in both bull and bear markets by following the trend.
# Target: 25-35 trades/year via strict EMA alignment + volume surge requirement.

name = "4h_ema_pullback_volume_surge_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # EMA21 for pullback target (calculated on 4h data)
    ema21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # EMA50 for trend filter (calculated on 4h data)
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # 20-period average volume for volume surge filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for higher timeframe trend filter
    df_d = get_htf_data(prices, '1d')
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Daily EMA50 for higher timeframe trend confirmation
    ema50_d = pd.Series(close_d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    bars_since_entry = 0  # Track bars since entry for minimum holding period
    
    # Start from sufficient lookback
    start_idx = max(50, 20)  # Need enough data for EMAs and volume average
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Get aligned daily EMA50 for trend filter
        ema50_d_val = align_htf_to_ltf(prices, df_d, ema50_d)[i]
        
        # Skip if any required data is NaN
        if (np.isnan(ema21[i]) or np.isnan(ema50[i]) or 
            np.isnan(ema50_d_val) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume surge condition: current volume > 2.5x 20-period average
        volume_surge = volume[i] > 2.5 * vol_ma[i]
        
        if position == 1:  # Long position
            bars_since_entry += 1
            # Exit if price breaks below EMA21 OR maximum holding period reached
            if close[i] < ema21[i] or bars_since_entry >= 6:
                position = 0
                signals[i] = 0.0
                bars_since_entry = 0
            elif position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            bars_since_entry += 1
            # Exit if price breaks above EMA21 OR maximum holding period reached
            if close[i] > ema21[i] or bars_since_entry >= 6:
                position = 0
                signals[i] = 0.0
                bars_since_entry = 0
            elif position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            bars_since_entry = 0
            # Pullback long: price crosses above EMA21 with volume surge, aligned with uptrend
            if (close[i] > ema21[i] and 
                close[i-1] <= ema21[i-1] and  # crossed above this bar
                volume_surge and 
                ema50[i] > ema50_d_val):  # 4h EMA50 above daily EMA50 = uptrend alignment
                position = 1
                signals[i] = 0.25
            # Pullback short: price crosses below EMA21 with volume surge, aligned with downtrend
            elif (close[i] < ema21[i] and 
                  close[i-1] >= ema21[i-1] and  # crossed below this bar
                  volume_surge and 
                  ema50[i] < ema50_d_val):  # 4h EMA50 below daily EMA50 = downtrend alignment
                position = -1
                signals[i] = -0.25
    
    return signals