#!/usr/bin/env python3
"""
Experiment #2111: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: Combining 6h Donchian breakouts with 1d weekly pivot levels (R3/S3 for fade, R4/S4 for breakout) 
and volume confirmation captures institutional order flow. Weekly pivots act as magnet levels - price tends to 
reverse at R3/S3 and accelerate through R4/S4. This structure works in both bull and bear markets by 
adapting to price action around key weekly levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2111_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from prior 1d data (using typical price)
    # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    # We'll use rolling 5-day approximation for weekly (1 trading week = 5 days)
    lookback = 5
    rolled_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    rolled_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    rolled_close = pd.Series(close_1d).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    # Weekly pivot levels
    pp = (rolled_high + rolled_low + rolled_close) / 3.0
    r1 = 2 * pp - rolled_low
    s1 = 2 * pp - rolled_high
    r2 = pp + (rolled_high - rolled_low)
    s2 = pp - (rolled_high - rolled_low)
    r3 = rolled_high + 2 * (pp - rolled_low)
    s3 = rolled_low - 2 * (rolled_high - pp)
    r4 = rolled_high + 3 * (pp - rolled_low)
    s4 = rolled_low - 3 * (rolled_high - pp)
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
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
            np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops below S3 (mean reversion at weekly support)
                if price < s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above R4 (take profit on breakout)
                elif price > r4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises above R3 (mean reversion at weekly resistance)
                if price > r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below S4 (take profit on breakdown)
                elif price < s4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND above R3 (breakout confirmation)
            if price > donchian_upper[i] and price > r3_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND below S3 (breakdown confirmation)
            elif price < donchian_lower[i] and price < s3_aligned[i]:
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