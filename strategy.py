#!/usr/bin/env python3
"""
Experiment #2191: 6h Donchian(20) breakout + 1d Camarilla pivot levels + volume confirmation
HYPOTHESIS: 6h Donchian breakouts with Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout) 
capture institutional order flow. Volume confirmation filters false breakouts. 
Works in bull/bear via pivot structure: mean revert at R3/S3, breakout at R4/S4.
Target: 75-150 total trades over 4 years (19-37/year) - optimized for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2191_6h_donchian20_1d_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (Range * 1.1/2)
    # R3 = C + (Range * 1.1/4)
    # S3 = C - (Range * 1.1/4)
    # S4 = C - (Range * 1.1/2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    r4_1d = close_1d + (range_1d * 1.1 / 2.0)
    r3_1d = close_1d + (range_1d * 1.1 / 4.0)
    s3_1d = close_1d - (range_1d * 1.1 / 4.0)
    s4_1d = close_1d - (range_1d * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 6h Indicators: Donchian(20), Volume MA(20) ===
    # Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
    # Volume MA for confirmation
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
            np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit conditions:
                # 1. Stoploss: 2*ATR equivalent (using 20-period range as proxy)
                price_range_20 = pd.Series(high - low).rolling(window=20, min_periods=1).iloc[i]
                if price < highest_since_entry - 1.5 * price_range_20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # 2. Take profit at R4
                elif price >= r4_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # 3. Mean reversion at R3 (fade)
                elif price >= r3_1d_aligned[i] and position_side > 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit conditions:
                # 1. Stoploss
                price_range_20 = pd.Series(high - low).rolling(window=20, min_periods=1).iloc[i]
                if price > lowest_since_entry + 1.5 * price_range_20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # 2. Take profit at S4
                elif price <= s4_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # 3. Mean reversion at S3 (fade)
                elif price <= s3_1d_aligned[i] and position_side < 0:
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
            # Long entry logic:
            # Breakout: price > R4 with volume
            # Fade: price < S3 with volume (expecting reversion to mean)
            if price > r4_1d_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif price < s3_1d_aligned[i]:
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