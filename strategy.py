#!/usr/bin/env python3
# 6h_Liquidity_Vacuum_Reversal
# Hypothesis: After extended moves, liquidity vacuums form where price moves with low volume, creating reversal opportunities.
# Uses 12h trend filter, volume imbalance detection, and price rejection at extremes.
# Works in bull/bear markets by fading exhausted moves during low-volume conditions.

name = "6h_Liquidity_Vacuum_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume imbalance: current volume vs 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = np.where(volume_ma > 0, volume / volume_ma, 1.0)
    
    # Price position in recent range (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    range_width = highest_high - lowest_low
    # Avoid division by zero
    range_width = np.where(range_width == 0, 1e-10, range_width)
    price_position = (close - lowest_low) / range_width  # 0 = at low, 1 = at high
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20-period calculations
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ratio[i]) or 
            np.isnan(price_position[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Liquidity vacuum conditions: low volume + price at extreme
        low_volume = volume_ratio[i] < 0.6  # Volume significantly below average
        at_high_extreme = price_position[i] > 0.85  # Near 20-period high
        at_low_extreme = price_position[i] < 0.15   # Near 20-period low
        
        # Price rejection signals: long wick at extreme
        body_size = abs(close[i] - prices['open'].iloc[i])
        total_range = high[i] - low[i]
        upper_wick = high[i] - max(close[i], prices['open'].iloc[i])
        lower_wick = min(close[i], prices['open'].iloc[i]) - low[i]
        
        # Avoid division by zero
        if total_range == 0:
            total_range = 1e-10
        upper_wick_ratio = upper_wick / total_range
        lower_wick_ratio = lower_wick / total_range
        
        rejection_at_high = upper_wick_ratio > 0.6  # Long upper wick
        rejection_at_low = lower_wick_ratio > 0.6   # Long lower wick
        
        if position == 0:
            # Long: price rejecting low extreme in uptrend with low volume (liquidity vacuum)
            if at_low_extreme and rejection_at_low and low_volume and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price rejecting high extreme in downtrend with low volume
            elif at_high_extreme and rejection_at_high and low_volume and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks above extreme or volume returns
            if price_position[i] > 0.95 or volume_ratio[i] > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks below extreme or volume returns
            if price_position[i] < 0.05 or volume_ratio[i] > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals