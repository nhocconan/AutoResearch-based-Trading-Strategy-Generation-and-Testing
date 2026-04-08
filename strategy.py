#!/usr/bin/env python3
# 6h_market_structure_reversal_v1
# Hypothesis: Identify reversal opportunities at higher timeframe support/resistance levels
# using 12h pivot points combined with 6h price action and volume confirmation.
# Works in bull/bear: pivot levels provide objective reference points, reversals
# occur at these levels regardless of trend direction. Volume confirms institutional interest.
# Target: 15-25 trades per year (60-100 total over 4 years).

name = "6h_market_structure_reversal_v1"
timeframe = "6h"
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
    
    # Calculate 12h pivot points (using prior 12h bar's OHLC)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate pivot points from previous 12h bar
    # Using standard formula: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    
    prev_high = df_12h['high'].shift(1).values  # Previous 12h high
    prev_low = df_12h['low'].shift(1).values    # Previous 12h low
    prev_close = df_12h['close'].shift(1).values # Previous 12h close
    
    # Calculate pivot levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)
    
    # Align pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_12h, pivot)
    r1_6h = align_htf_to_ltf(prices, df_12h, r1)
    s1_6h = align_htf_to_ltf(prices, df_12h, s1)
    r2_6h = align_htf_to_ltf(prices, df_12h, r2)
    s2_6h = align_htf_to_ltf(prices, df_12h, s2)
    r3_6h = align_htf_to_ltf(prices, df_12h, r3)
    s3_6h = align_htf_to_ltf(prices, df_12h, s3)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_ma[20:] = pd.Series(volume).rolling(window=20, min_periods=20).mean()[20:].values
    
    # Price position relative to pivot levels
    # Calculate distances to nearest support/resistance
    dist_to_r1 = np.abs(close - r1_6h)
    dist_to_s1 = np.abs(close - s1_6h)
    dist_to_r2 = np.abs(close - r2_6h)
    dist_to_s2 = np.abs(close - s2_6h)
    dist_to_r3 = np.abs(close - r3_6h)
    dist_to_s3 = np.abs(close - s3_6h)
    
    # Find minimum distance to any pivot level
    min_dist = np.minimum.reduce([dist_to_r1, dist_to_s1, dist_to_r2, dist_to_s2, dist_to_r3, dist_to_s3])
    
    # Entry condition: price near pivot level (within 0.3% of price)
    proximity_threshold = 0.003  # 0.3%
    near_pivot = min_dist / close < proximity_threshold
    
    # Determine if we're at resistance or support for direction
    # At resistance: look for rejection (price failing to break higher)
    # At support: look for bounce (price finding support)
    
    # Check for rejection at resistance levels
    at_resistance = (
        (np.abs(close - r1_6h) / close < proximity_threshold) |
        (np.abs(close - r2_6h) / close < proximity_threshold) |
        (np.abs(close - r3_6h) / close < proximity_threshold)
    )
    
    # Check for bounce at support levels
    at_support = (
        (np.abs(close - s1_6h) / close < proximity_threshold) |
        (np.abs(close - s2_6h) / close < proximity_threshold) |
        (np.abs(close - s3_6h) / close < proximity_threshold)
    )
    
    # Volume confirmation
    volume_filter = volume > 1.3 * vol_ma
    
    # Price action confirmation: look for rejection/bounce patterns
    # For resistance: current close below open AND near resistance
    # For support: current close above open AND near support
    bearish_rejection = (close < prices['open'].values) & at_resistance
    bullish_bounce = (close > prices['open'].values) & at_support
    
    # Start from sufficient lookback
    start_idx = 30
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or
            np.isnan(s3_6h[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        vol_filter = volume_filter[i]
        
        if position == 1:  # Long position
            # Exit if price breaks below support or reversal signal
            if close[i] < s1_6h[i] or bearish_rejection[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above resistance or reversal signal
            if close[i] > r1_6h[i] or bullish_bounce[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Short entry: bearish rejection at resistance with volume
            if bearish_rejection[i] and vol_filter:
                position = -1
                signals[i] = -0.25
            # Long entry: bullish bounce at support with volume
            elif bullish_bounce[i] and vol_filter:
                position = 1
                signals[i] = 0.25
    
    return signals