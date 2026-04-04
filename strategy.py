#!/usr/bin/env python3
"""
Experiment #2731: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot levels (from 1d data) capture
institutional participation while avoiding false breakouts. Weekly pivot provides structural
bias (R4/S4 for continuation, R3/S3 for mean reversion) that works in both bull and bear markets.
Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2731_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot points from prior week (using daily OHLC)
    # Weekly high/low/close from prior completed week
    # We'll use rolling window of 5 trading days (approximation for weekly)
    if len(close_1d) >= 5:
        weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values  # shift(1) for prior week
        weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
        weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(1).values
        
        # Weekly pivot point: (H + L + C) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        
        # Weekly support/resistance levels
        r1 = 2 * weekly_pivot - weekly_low
        s1 = 2 * weekly_pivot - weekly_high
        r2 = weekly_pivot + (weekly_high - weekly_low)
        s2 = weekly_pivot - (weekly_high - weekly_low)
        r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
        s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
        r4 = weekly_high + 3 * (weekly_pivot - weekly_low)
        s4 = weekly_low - 3 * (weekly_high - weekly_pivot)
    else:
        # Not enough data for weekly calculation
        weekly_pivot = np.full_like(close_1d, np.nan)
        r1 = np.full_like(close_1d, np.nan)
        s1 = np.full_like(close_1d, np.nan)
        r2 = np.full_like(close_1d, np.nan)
        s2 = np.full_like(close_1d, np.nan)
        r3 = np.full_like(close_1d, np.nan)
        s3 = np.full_like(close_1d, np.nan)
        r4 = np.full_like(close_1d, np.nan)
        s4 = np.full_like(close_1d, np.nan)
    
    # Align weekly pivot levels to 6h timeframe (with shift(1) for completed weekly bars only)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators: Donchian(20) channels, Volume MA(20) ===
    # Donchian channels (20-period high/low)
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
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
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
                # Exit if price drops below weekly S3 (mean reversion from extreme)
                if price < s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian low (failed breakout)
                elif price < lowest_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises above weekly R3 (mean reversion from extreme)
                if price > r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian high (failed breakout)
                elif price > highest_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.8x average to reduce frequency)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Long entry: price breaks above Donchian high AND above weekly R4 (strong continuation)
            if price > highest_20[i] and price > r4_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low AND below weekly S4 (strong continuation)
            elif price < lowest_20[i] and price < s4_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            # Long mean reversion: price breaks below Donchian low but above weekly S3 (oversold bounce)
            elif price < lowest_20[i] and price > s3_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short mean reversion: price breaks above Donchian high but below weekly R3 (overbought rejection)
            elif price > highest_20[i] and price < r3_aligned[i]:
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