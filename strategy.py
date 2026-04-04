#!/usr/bin/env python3
"""
Experiment #2411: 6h Donchian(20) breakout + 1d Weekly Pivot + Volume Confirmation
HYPOTHESIS: Combining 6h Donchian breakouts with 1d weekly pivot levels (R3/S3 for fade, R4/S4 for continuation) 
and volume confirmation captures institutional participation during key level breaks. Weekly pivots provide 
structure from larger timeframe, reducing false breakouts. Works in bull markets (continuation at R4/S4) 
and bear markets (fade at R3/S3). Discrete position sizing (0.25) limits fee drag. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2411_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # We'll use rolling window of 5 days (1 week) to get weekly high/low/close
    if len(high_1d) >= 5:
        weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
        weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
        weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
        
        # Pivot point = (H + L + C) / 3
        pp = (weekly_high + weekly_low + weekly_close) / 3.0
        
        # Support and resistance levels
        r1 = 2 * pp - weekly_low
        s1 = 2 * pp - weekly_high
        r2 = pp + (weekly_high - weekly_low)
        s2 = pp - (weekly_high - weekly_low)
        r3 = weekly_high + 2 * (pp - weekly_low)
        s3 = weekly_low - 2 * (weekly_high - pp)
        r4 = r3 + (r2 - r1)
        s4 = s3 - (s2 - s1)
        
        # Align to 6h timeframe
        pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        # Not enough data for weekly calculation
        pp_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
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
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
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
                # Exit if price breaks below Donchian low (mean reversion)
                elif price < lowest_20[i]:
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
                # Exit if price breaks above Donchian high (mean reversion)
                elif price > highest_20[i]:
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
            # Fade at R3/S3 (mean reversion at strong weekly levels)
            if price >= r3_aligned[i] and price <= r4_aligned[i]:
                # In R3-R4 zone: short on rejection from R3
                if price < (r3_aligned[i] + r4_aligned[i]) / 2:  # Below midpoint of R3-R4
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE
            elif price >= s4_aligned[i] and price <= s3_aligned[i]:
                # In S4-S3 zone: long on rejection from S3
                if price > (s3_aligned[i] + s4_aligned[i]) / 2:  # Above midpoint of S3-S4
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = SIZE
            # Breakout continuation at R4/S4 (break through strong weekly levels)
            elif price > r4_aligned[i]:
                # Break above R4: long continuation
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif price < s4_aligned[i]:
                # Break below S4: short continuation
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