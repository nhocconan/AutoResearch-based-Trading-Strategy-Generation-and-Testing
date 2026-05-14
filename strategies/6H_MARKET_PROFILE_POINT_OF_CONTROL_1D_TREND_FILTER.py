#!/usr/bin/env python3
# 6H_MARKET_PROFILE_POINT_OF_CONTROL_1D_TREND_FILTER
# Hypothesis: Use daily Value Area High/Low from Market Profile to identify institutional value areas.
# Enter long when price breaks above Value Area High with volume confirmation in bullish regime,
# enter short when price breaks below Value Area Low with volume confirmation in bearish regime.
# Uses Point of Control (POC) as dynamic support/resistance for exits.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drift while capturing institutional flow.

name = "6H_MARKET_PROFILE_POINT_OF_CONTROL_1D_TREND_FILTER"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Market Profile and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for VA calculation
        return np.zeros(n)
    
    # Calculate Value Area and Point of Control using daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Initialize arrays for VAH, VAL, POC
    vah = np.full(len(df_1d), np.nan)
    val = np.full(len(df_1d), np.nan)
    poc = np.full(len(df_1d), np.nan)
    
    # Calculate Value Area (70% of volume) and Point of Control for each day
    for i in range(len(df_1d)):
        # Create price profile for the day using high-low range
        price_range = high_1d[i] - low_1d[i]
        if price_range <= 0:
            continue
            
        # Divide range into 30 price bins (standard Market Profile)
        bins = 30
        bin_size = price_range / bins
        price_bins = low_1d[i] + np.arange(bins) * bin_size
        
        # Volume distribution - simplified: assume volume distributed across range
        # In reality would need TPO data, but we approximate with price action
        vol_dist = np.full(bins, volume_1d[i] / bins)
        
        # Point of Control = price bin with maximum volume
        poc_idx = np.argmax(vol_dist)
        poc[i] = price_bins[poc_idx]
        
        # Value Area = range containing 70% of volume around POC
        vol_cumsum = np.cumsum(vol_dist)
        total_vol = vol_cumsum[-1]
        va_low_idx = 0
        va_high_idx = bins - 1
        
        # Find range that contains ~70% of volume centered on POC
        target_vol = 0.7 * total_vol
        for j in range(bins):
            for k in range(j, bins):
                vol_in_range = vol_cumsum[k] - (vol_cumsum[j-1] if j > 0 else 0)
                if abs(vol_in_range - target_vol) < 0.05 * total_vol:  # Within 5% tolerance
                    if k - j < va_high_idx - va_low_idx:  # Prefer narrower range
                        va_low_idx = j
                        va_high_idx = k
        
        val[i] = price_bins[va_low_idx]
        vah[i] = price_bins[va_high_idx]
    
    # Alternative simpler calculation if above fails (fallback to standard method)
    # Use typical price approximation for Value Area
    typical_price = (high_1d + low_1d + close_1d) / 3
    
    # If VA calculation failed, use standard deviation bands
    if np.all(np.isnan(vah)):
        tp_mean = np.mean(typical_price)
        tp_std = np.std(typical_price)
        vah = tp_mean + 0.7 * tp_std
        val = tp_mean - 0.7 * tp_std
        poc = tp_mean
    
    # 1-day EMA for trend filter (34-period)
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Market Profile levels to 6h timeframe
    vah_aligned = align_htf_to_ltf(prices, df_1d, vah)
    val_aligned = align_htf_to_ltf(prices, df_1d, val)
    poc_aligned = align_htf_to_ltf(prices, df_1d, poc)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Volume spike detection (24-period volume MA on 6h - approx 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > vol_ma * 2.0  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(vah_aligned[i]) or np.isnan(val_aligned[i]) or 
            np.isnan(poc_aligned[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Value Area High with volume confirmation in bullish regime
            if (close[i] > vah_aligned[i] and vol_spike[i] and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Value Area Low with volume confirmation in bearish regime
            elif (close[i] < val_aligned[i] and vol_spike[i] and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to Point of Control (intraday mean reversion)
            if close[i] < poc_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to Point of Control
            if close[i] > poc_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals