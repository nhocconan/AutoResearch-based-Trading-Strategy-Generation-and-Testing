#!/usr/bin/env python3
"""
12h_market_profile_value_area_1d_trend_volume_v1
Hypothesis: On 12h timeframe, trade mean reversion from Value Area (VA) High/Low 
calculated from prior day's TPO (Time Price Opportunity) profile, with trend filter 
from daily EMA and volume confirmation. In ranging markets, price tends to revert 
to the Point of Control (POC). In trending markets, trade pullbacks to VA in 
direction of trend. Volume confirms institutional interest at key levels.
Works in both bull and bear markets by adapting to trend via EMA filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_market_profile_value_area_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def calculate_tpo_profile(high, low, close, tick_size=0.1):
    """
    Calculate Point of Control (POC) and Value Area (VA) from TPO profile.
    Simplified: assumes uniform distribution within each bar's range.
    Returns POC price and VA High/Low (70% of volume around POC).
    """
    # For each bar, create a simple price distribution
    # In practice, this would use actual TPO data, but we approximate
    # using the bar's range divided into price levels
    
    # Use a fixed number of price levels per bar for simplicity
    n_levels = 10
    
    poc_values = []
    va_high_values = []
    va_low_values = []
    
    for h, l, c in zip(high, low, close):
        if h == l:  # Avoid division by zero
            poc_values.append(c)
            va_high_values.append(c)
            va_low_values.append(c)
            continue
            
        # Create price levels from low to high
        price_levels = np.linspace(l, h, n_levels)
        
        # Simple TPO: each level gets 1 unit of time
        # In reality, would weight by actual time spent at each price
        tpo_counts = np.ones(n_levels)
        
        # Find POC (price level with highest TPO)
        poc_idx = np.argmax(tpo_counts)
        poc = price_levels[poc_idx]
        
        # Calculate Value Area (70% of TPO around POC)
        total_tpo = np.sum(tpo_counts)
        target_tpo = 0.7 * total_tpo
        
        # Expand out from POC to find VA boundaries
        cum_tpo = tpo_counts[poc_idx]
        va_low_idx = poc_idx
        va_high_idx = poc_idx
        
        while cum_tpo < target_tpo and (va_low_idx > 0 or va_high_idx < n_levels - 1):
            # Expand to the side with more TPO
            left_tpo = tpo_counts[va_low_idx - 1] if va_low_idx > 0 else 0
            right_tpo = tpo_counts[va_high_idx + 1] if va_high_idx < n_levels - 1 else 0
            
            if left_tpo >= right_tpo and va_low_idx > 0:
                va_low_idx -= 1
                cum_tpo += tpo_counts[va_low_idx]
            elif right_tpo > left_tpo and va_high_idx < n_levels - 1:
                va_high_idx += 1
                cum_tpo += tpo_counts[va_high_idx]
            elif va_low_idx > 0:
                va_low_idx -= 1
                cum_tpo += tpo_counts[va_low_idx]
            elif va_high_idx < n_levels - 1:
                va_high_idx += 1
                cum_tpo += tpo_counts[va_high_idx]
            else:
                break
        
        va_low = price_levels[va_low_idx]
        va_high = price_levels[va_high_idx]
        
        poc_values.append(poc)
        va_high_values.append(va_high)
        va_low_values.append(va_low)
    
    return np.array(poc_values), np.array(va_high_values), np.array(va_low_values)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Market Profile and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC for Market Profile
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Point of Control (POC) and Value Area (VA) for each day
    poc_1d, va_high_1d, va_low_1d = calculate_tpo_profile(high_1d, low_1d, close_1d)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # Align daily levels to 12h timeframe
    poc_12h = align_htf_to_ltf(prices, df_1d, poc_1d)
    va_high_12h = align_htf_to_ltf(prices, df_1d, va_high_1d)
    va_low_12h = align_htf_to_ltf(prices, df_1d, va_low_1d)
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 20-period volume average on 12h
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(poc_12h[i]) or np.isnan(va_high_12h[i]) or 
            np.isnan(va_low_12h[i]) or np.isnan(ema50_12h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below VA Low OR price crosses below EMA in downtrend
            if close[i] < va_low_12h[i] or (close[i] < ema50_12h[i] and ema50_12h[i] < va_low_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above VA High OR price crosses above EMA in uptrend
            if close[i] > va_high_12h[i] or (close[i] > ema50_12h[i] and ema50_12h[i] > va_high_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reversion longs at VA Low in uptrend (price > EMA)
            if (close[i] <= va_low_12h[i] * 1.001 and  # Small buffer to avoid exact equality
                vol_confirm and 
                close[i] > ema50_12h[i]):
                position = 1
                signals[i] = 0.25
            # Mean reversion shorts at VA High in downtrend (price < EMA)
            elif (close[i] >= va_high_12h[i] * 0.999 and  # Small buffer to avoid exact equality
                  vol_confirm and 
                  close[i] < ema50_12h[i]):
                position = -1
                signals[i] = -0.25
            # Trend continuation longs pulling back to VA in uptrend
            elif (close[i] >= va_low_12h[i] and 
                  close[i] <= va_high_12h[i] and  # Inside VA
                  vol_confirm and 
                  close[i] > ema50_12h[i] and
                  close[i] > close[i-1]):  # Price rising
                position = 1
                signals[i] = 0.20  # Smaller size for trend continuation
            # Trend continuation shorts pulling back to VA in downtrend
            elif (close[i] <= va_high_12h[i] and 
                  close[i] >= va_low_12h[i] and  # Inside VA
                  vol_confirm and 
                  close[i] < ema50_12h[i] and
                  close[i] < close[i-1]):  # Price falling
                position = -1
                signals[i] = -0.20  # Smaller size for trend continuation
    
    return signals