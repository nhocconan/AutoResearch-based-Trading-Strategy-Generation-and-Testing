#!/usr/bin/env python3
"""
1d_Daily_Volume_Profile_POC_With_Trend_Filter
Hypothesis: Uses Point of Control (POC) from daily volume profile as dynamic support/resistance.
Long when price > POC and above weekly EMA50 (bullish trend), short when price < POC and below weekly EMA50 (bearish trend).
Requires volume confirmation (>1.5x 20-day average) to filter low-probability breakouts.
Designed for low-frequency (~15 trades/year) with strong edge in both bull and bear markets via trend alignment.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume profile POC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily POC (Point of Control) - price level with highest volume
    poc = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        # Use last 20 days to build volume profile (more stable)
        start_idx = max(0, i - 19)
        end_idx = i + 1
        
        # Create price bins for volume profile
        price_slice = high[start_idx:end_idx]
        low_slice = low[start_idx:end_idx]
        vol_slice = volume[start_idx:end_idx]
        
        if len(price_slice) == 0:
            continue
            
        # Define price range and bins
        price_min = np.min(low_slice)
        price_max = np.max(price_slice)
        if price_max <= price_min:
            continue
            
        bins = 50
        bin_edges = np.linspace(price_min, price_max, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Volume distribution across price bins
        vol_dist = np.zeros(bins)
        for j in range(len(price_slice)):
            # Find which bin the typical price falls into
            typical_price = (price_slice[j] + low_slice[j]) / 2
            if typical_price < price_min or typical_price > price_max:
                continue
            bin_idx = np.searchsorted(bin_edges, typical_price) - 1
            if 0 <= bin_idx < bins:
                vol_dist[bin_idx] += vol_slice[j]
        
        # POC is bin center with maximum volume
        if np.max(vol_dist) > 0:
            poc_idx = np.argmax(vol_dist)
            poc[i] = bin_centers[poc_idx]
    
    # Align POC to 1d timeframe (already daily, but align for safety)
    poc_aligned = align_htf_to_ltf(prices, df_1d, poc)
    
    # Volume confirmation: >1.5x 20-day average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(poc_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation (>1.5x average)
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry conditions: price crosses POC with trend and volume
        long_signal = (close[i] > poc_aligned[i]) and uptrend and vol_confirm
        short_signal = (close[i] < poc_aligned[i]) and downtrend and vol_confirm
        
        # Exit conditions: price returns to POC
        long_exit = close[i] < poc_aligned[i]
        short_exit = close[i] > poc_aligned[i]
        
        if long_signal and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Daily_Volume_Profile_POC_With_Trend_Filter"
timeframe = "1d"
leverage = 1.0