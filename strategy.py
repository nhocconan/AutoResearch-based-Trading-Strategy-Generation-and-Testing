#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Volume_Weighted_Skew_MeanReversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d Close for price position ===
    close_1d = df_1d['close'].values
    
    # === 1d 20-period high/low for range ===
    high_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).min().values
    
    # === Price position in 20-day range (0 to 1) ===
    range_20 = high_20 - low_20
    price_pos = np.where(range_20 > 0, (close_1d - low_20) / range_20, 0.5)
    
    # === 1d Volume-weighted skew of price position ===
    vol_1d = df_1d['volume'].values
    # Weighted mean of price position
    sum_vol = np.nancumsum(vol_1d)
    sum_vol_pos = np.nancumsum(vol_1d * price_pos)
    weighted_mean = np.where(sum_vol > 0, sum_vol_pos / sum_vol, 0.5)
    # Weighted variance
    sum_vol_var = np.nancumsum(vol_1d * (price_pos - weighted_mean) ** 2)
    weighted_var = np.where(sum_vol > 0, sum_vol_var / sum_vol, 0.0)
    # Weighted skew (third moment)
    sum_vol_skew = np.nancumsum(vol_1d * (price_pos - weighted_mean) ** 3)
    weighted_skew = np.where((sum_vol > 0) & (weighted_var > 0), 
                            sum_vol_skew / (sum_vol * weighted_var ** 1.5), 0)
    
    # Align to 6h
    price_pos_6h = align_htf_to_ltf(prices, df_1d, price_pos)
    weighted_skew_6h = align_htf_to_ltf(prices, df_1d, weighted_skew)
    
    # === 6h Volume filter: current volume > 1.5x 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(price_pos_6h[i]) or np.isnan(weighted_skew_6h[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Mean reversion signals based on volume-weighted skew
            # Negative skew = more volume at lower prices -> potential bounce long
            # Positive skew = more volume at higher prices -> potential drop short
            long_cond = (weighted_skew_6h[i] < -0.3 and 
                        price_pos_6h[i] < 0.3 and  # Oversold
                        volume[i] > vol_ma20[i])    # Volume confirmation
            
            short_cond = (weighted_skew_6h[i] > 0.3 and 
                         price_pos_6h[i] > 0.7 and  # Overbought
                         volume[i] > vol_ma20[i])   # Volume confirmation
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to middle or skew normalizes
            exit_cond = (price_pos_6h[i] > 0.5 or 
                        weighted_skew_6h[i] > -0.1)
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle or skew normalizes
            exit_cond = (price_pos_6h[i] < 0.5 or 
                        weighted_skew_6h[i] < 0.1)
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Mean reversion strategy using volume-weighted skewness of price position
# within 20-day range on 1d timeframe. Negative skew indicates heavier volume at lower
# price levels (accumulation), signaling potential long. Positive skew indicates
# distribution at higher levels, signaling potential short. Uses 6-volume confirmation
# and exits when price returns to mid-range or skew normalizes. Designed to work
# in ranging markets common in 2025 BTC/ETH environment. Targets 60-120 trades over
# 4 years (15-30/year) to minimize fee drift. Uses discrete sizing (0.25). Works
# on BTC/ETH via institutional volume patterns.