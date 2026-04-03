#!/usr/bin/env python3
"""
Experiment #2019: 6h Donchian(20) breakout + 12h volume spike + 1d Camarilla pivot fade/continuation
HYPOTHESIS: 
- Primary: 6h Donchian(20) breakout with volume > 2.0x 20-bar 12h average (institutional participation)
- HTF: 1d Camarilla pivot levels - fade at R3/S3 (mean reversion in ranges), breakout continuation at R4/S4 (trend)
- Works in bull/bear by using 1d pivot structure to determine whether to fade or follow breakouts
- Volume spike filter ensures only high-confidence institutional breakouts
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2019_6h_donchian20_12h_vol_1d_camarilla_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume confirmation (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Volume MA(20) on 12h for spike detection
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = np.ones(len(close_12h))
    vol_ratio_12h[20:] = volume_12h[20:] / vol_ma_12h[20:]
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = Pivot + (Range * 1.1/2)
    # R3 = Pivot + (Range * 1.1/4)
    # S3 = Pivot - (Range * 1.1/4)
    # S4 = Pivot - (Range * 1.1/2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    r4_1d = pivot_1d + (range_1d * 1.1 / 2)
    r3_1d = pivot_1d + (range_1d * 1.1 / 4)
    s3_1d = pivot_1d - (range_1d * 1.1 / 4)
    s4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 6h Indicators: Donchian(20) ===
    # Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
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
            np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (calculated on 6h)
                tr = np.maximum(high[i] - low[i], 
                               np.maximum(np.abs(high[i] - close[i-1]), 
                                          np.abs(low[i] - close[i-1]))) if i > 0 else high[i] - low[i]
                # Simplified ATR approximation for exit
                if price < highest_since_entry - 2.5 * (high[i] - low[i]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches opposite Camarilla level (mean reversion)
                elif price <= s3_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                tr = np.maximum(high[i] - low[i], 
                               np.maximum(np.abs(high[i] - close[i-1]), 
                                          np.abs(low[i] - close[i-1]))) if i > 0 else high[i] - low[i]
                if price > lowest_since_entry + 2.5 * (high[i] - low[i]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches opposite Camarilla level (mean reversion)
                elif price >= r3_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 2.0x average on 12h)
        volume_spike = vol_ratio_12h_aligned[i] > 2.0
        
        if volume_spike:
            # Determine market regime based on price vs Camarilla levels
            # In range (between S3 and R3): fade breaks at R3/S3
            # In trend (above R4 or below S4): continue breaks at R4/S4
            
            # Long entry conditions
            if price > donchian_upper[i]:
                if (s3_1d_aligned[i] <= price <= r3_1d_aligned[i]):  # In range - fade at R3
                    if price <= r3_1d_aligned[i] and price > s3_1d_aligned[i]:
                        # Fade long when price rejects R3 (mean reversion)
                        if price < r3_1d_aligned[i] and low[i] < r3_1d_aligned[i]:
                            # Look for rejection wick at R3
                            if (close[i] - low[i]) > 0.6 * (high[i] - low[i]):  # Long lower wick
                                in_position = True
                                position_side = 1
                                entry_price = close[i]
                                highest_since_entry = high[i]
                                lowest_since_entry = low[i]
                                signals[i] = SIZE
                elif price >= r4_1d_aligned[i]:  # In trend - continue at R4 breakout
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = SIZE
            
            # Short entry conditions
            elif price < donchian_lower[i]:
                if (s3_1d_aligned[i] <= price <= r3_1d_aligned[i]):  # In range - fade at S3
                    if price >= s3_1d_aligned[i] and price < r3_1d_aligned[i]:
                        # Fade short when price rejects S3 (mean reversion)
                        if price > s3_1d_aligned[i] and high[i] > s3_1d_aligned[i]:
                            # Look for rejection wick at S3
                            if (high[i] - close[i]) > 0.6 * (high[i] - low[i]):  # Long upper wick
                                in_position = True
                                position_side = -1
                                entry_price = close[i]
                                highest_since_entry = high[i]
                                lowest_since_entry = low[i]
                                signals[i] = -SIZE
                elif price <= s4_1d_aligned[i]:  # In trend - continue at S4 breakdown
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals