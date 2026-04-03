#!/usr/bin/env python3
"""
Experiment #2215: 6h Donchian(20) breakout + 1w Camarilla pivot levels + volume confirmation
HYPOTHESIS: 6h Donchian breakouts with 1w Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
and volume confirmation work in both bull and bear markets. Camarilla levels provide institutional 
support/resistance that holds across regimes, while Donchian breakouts capture momentum. Volume 
filter ensures only significant breakouts are traded. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2215_6h_donchian20_1w_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivot levels (based on previous week)
    # Pivot = (H + L + C) / 3
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    pivot_1w = np.full_like(close_1w, np.nan)
    r4_1w = np.full_like(close_1w, np.nan)
    r3_1w = np.full_like(close_1w, np.nan)
    s3_1w = np.full_like(close_1w, np.nan)
    s4_1w = np.full_like(close_1w, np.nan)
    
    for i in range(1, len(close_1w)):
        if not (np.isnan(high_1w[i-1]) or np.isnan(low_1w[i-1]) or np.isnan(close_1w[i-1])):
            pivot_1w[i] = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3.0
            range_1w = high_1w[i-1] - low_1w[i-1]
            r4_1w[i] = close_1w[i-1] + (range_1w * 1.1 / 2)
            r3_1w[i] = close_1w[i-1] + (range_1w * 1.1 / 4)
            s3_1w[i] = close_1w[i-1] - (range_1w * 1.1 / 4)
            s4_1w[i] = close_1w[i-1] - (range_1w * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 for completed bars)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # === 6h Indicators: Donchian(20), Volume MA(20) ===
    # Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(pivot_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or
            np.isnan(r4_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(s4_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing logic
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops below S3 (mean reversion to support)
                if price < s3_1w_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches upper Donchian (take profit at channel top)
                elif price >= donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises above R3 (mean reversion to resistance)
                if price > r3_1w_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches lower Donchian (take profit at channel bottom)
                elif price <= donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above upper Donchian AND above R3 (bullish breakout)
            if price > donchian_upper[i] and price > r3_1w_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND below S3 (bearish breakout)
            elif price < donchian_lower[i] and price < s3_1w_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            # Long mean reversion: price touches S3 AND bounces off lower Donchian
            elif abs(price - s3_1w_aligned[i]) < 0.001 * price and price > donchian_lower[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short mean reversion: price touches R3 AND rejects off upper Donchian
            elif abs(price - r3_1w_aligned[i]) < 0.001 * price and price < donchian_upper[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals