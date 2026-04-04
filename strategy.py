#!/usr/bin/env python3
"""
Experiment #2407: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: Combining 6h Donchian breakouts with 1d weekly pivot levels (R4/S4 for continuation, R3/S3 for mean reversion) 
and volume confirmation captures institutional participation. Weekly pivots provide structural support/resistance that 
works in both bull (breakouts at R4/S4 with volume) and bear (reversals at R3/S3 with volume) markets. 
Discrete position sizing (0.25) limits fee drag while targeting 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2407_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from prior week's OHLC
    # Using rolling window of 5 trading days (1 week)
    if len(high_1d) >= 5:
        prev_week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values
        prev_week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
        prev_week_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(1).values
        
        pivot_point = (prev_week_high + prev_week_low + prev_week_close) / 3.0
        r1 = 2 * pivot_point - prev_week_low
        s1 = 2 * pivot_point - prev_week_high
        r2 = pivot_point + (prev_week_high - prev_week_low)
        s2 = pivot_point - (prev_week_high - prev_week_low)
        r3 = prev_week_high + 2 * (pivot_point - prev_week_low)
        s3 = prev_week_low - 2 * (prev_week_high - pivot_point)
        r4 = prev_week_high + 3 * (pivot_point - prev_week_low)
        s4 = prev_week_low - 3 * (prev_week_high - pivot_point)
    else:
        # Not enough data for weekly calculation
        pivot_point = r1 = s1 = r2 = s2 = r3 = s3 = r4 = s4 = np.full_like(close_1d, np.nan)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators: Donchian(20) channels, Volume MA(20) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry (using Donchian width as ATR proxy)
                donchian_width = highest_20[i] - lowest_20[i]
                atr_estimate = donchian_width * 0.15  # approximate ATR from channel width
                if price < highest_since_entry - 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below S3 (strong support) or above R4 (extended)
                elif price < s3_aligned[i] or price > r4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                donchian_width = highest_20[i] - lowest_20[i]
                atr_estimate = donchian_width * 0.15
                if price > lowest_since_entry + 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above R3 (strong resistance) or below S4 (extended)
                elif price > r3_aligned[i] or price < s4_aligned[i]:
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
            # Long entry: price breaks above Donchian high AND above R4 (continuation)
            # OR price breaks above Donchian high AND bounces from S3 (mean reversion)
            if price > highest_20[i]:
                if price > r4_aligned[i]:  # Breakout continuation at R4
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = SIZE
                elif price > s3_aligned[i] and price < r3_aligned[i]:  # Mean reversion from S3
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = SIZE
            # Short entry: price breaks below Donchian low AND below S4 (continuation)
            # OR price breaks below Donchian low AND rejected at R3 (mean reversion)
            elif price < lowest_20[i]:
                if price < s4_aligned[i]:  # Breakdown continuation at S4
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE
                elif price < r3_aligned[i] and price > s3_aligned[i]:  # Mean reversion from R3
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