# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_1d1w_Pivot_Reversal_With_Volume
Hypothesis: In BTC/ETH, weekly pivot points act as strong support/resistance in both bull and bear markets.
At weekly pivot levels, price often reverses with exhaustion (low volume) or breaks with conviction (high volume).
We fade reversals at S1/R1 (mean reversion) and breakout at S2/R2/R3/S3 (trend continuation).
Volume filter ensures we only trade when participation confirms the move.
Timeframe: 6s balances noise and signal quality, targeting 20-40 trades/year.
"""

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
    
    # Daily data for pivot calculation (needs daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Weekly data for higher timeframe bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week's OHLC
    # Using (H+L+C)/3 for pivot, then support/resistance levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point (PP) = (H + L + C) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # Support and resistance levels
    r1_1w = 2 * pp_1w - low_1w
    s1_1w = 2 * pp_1w - high_1w
    r2_1w = pp_1w + (high_1w - low_1w)
    s2_1w = pp_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pp_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pp_1w)
    
    # Align weekly pivot levels to 6s timeframe (wait for weekly close)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Volume confirmation: 6s volume vs 20-period average of 6s volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position
    
    for i in range(20, n):
        # Skip if pivot data not ready
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        volume_condition = volume[i] > (volume_ma_20[i] * 1.5)
        
        # Define zones around pivot levels
        pp = pp_1w_aligned[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        r2 = r2_1w_aligned[i]
        s2 = s2_1w_aligned[i]
        r3 = r3_1w_aligned[i]
        s3 = s3_1w_aligned[i]
        
        price = close[i]
        
        # Fade logic: reverse at S1/R1 when price approaches with low momentum
        # Breakout logic: break S2/R2 or S3/R3 with volume
        
        if position == 0:
            # Fade at S1/R1 (mean reversion)
            if price <= s1 and price >= s1 * 0.998:  # Near S1
                if volume_condition:
                    # Only fade if weekly bias is not strongly opposing
                    weekly_bias_up = close_1w[i-1] > pp_1w[i-1] if i > 0 and not np.isnan(close_1w[i-1]) and not np.isnan(pp_1w[i-1]) else True
                    if not weekly_bias_up:  # Fade only in down weekly bias or neutral
                        position = 1
                        signals[i] = position_size
            elif price >= r1 and price <= r1 * 1.002:  # Near R1
                if volume_condition:
                    weekly_bias_down = close_1w[i-1] < pp_1w[i-1] if i > 0 and not np.isnan(close_1w[i-1]) and not np.isnan(pp_1w[i-1]) else True
                    if not weekly_bias_down:  # Fade only in up weekly bias or neutral
                        position = -1
                        signals[i] = -position_size
            
            # Breakout at S2/R2 or S3/R3 with volume
            elif price < s2 and volume_condition:
                position = -1
                signals[i] = -position_size
            elif price > r2 and volume_condition:
                position = 1
                signals[i] = position_size
            elif price < s3 and volume_condition:
                position = -1
                signals[i] = -position_size * 0.5  # Smaller size for extreme break
            elif price > r3 and volume_condition:
                position = 1
                signals[i] = position_size * 0.5
        
        elif position == 1:
            # Exit long: price returns to pivot or breaks S1 with volume
            if price >= pp or (price <= s1 and volume_condition):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        
        elif position == -1:
            # Exit short: price returns to pivot or breaks R1 with volume
            if price <= pp or (price >= r1 and volume_condition):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d1w_Pivot_Reversal_With_Volume"
timeframe = "6h"
leverage = 1.0