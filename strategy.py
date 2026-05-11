#!/usr/bin/env python3
name = "6h_PremiumDiscount_Zone"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtr_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for range calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-day high-low range
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    range_20 = highest_20 - lowest_20
    
    # Calculate premium/discount zone
    # Premium zone: upper 40% of range (short bias)
    # Discount zone: lower 40% of range (long bias)
    # Middle 20%: no trade
    premium_zone = lowest_20 + 0.6 * range_20  # Start of upper 40%
    discount_zone = lowest_20 + 0.4 * range_20  # End of lower 40%
    
    # Align to 6h timeframe
    premium_zone_aligned = align_htf_to_ltf(prices, df_1d, premium_zone)
    discount_zone_aligned = align_htf_to_ltf(prices, df_1d, discount_zone)
    
    # Volume confirmation (20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Wait for 20-day range calculation
    
    for i in range(start_idx, n):
        if np.isnan(premium_zone_aligned[i]) or np.isnan(discount_zone_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price in discount zone + volume spike
            if close[i] <= discount_zone_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price in premium zone + volume spike
            elif close[i] >= premium_zone_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price moves to middle or premium zone
            if close[i] > discount_zone_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price moves to middle or discount zone
            if close[i] < premium_zone_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals