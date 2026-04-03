#!/usr/bin/env python3
"""
Experiment #2147: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: Donchian channel breakouts on 6h timeframe capture intermediate-term momentum with daily trend filter via weekly pivot levels. Weekly pivot (calculated from prior week) provides structural support/resistance that works in both bull (breakout continuation) and bear (fade at resistance) markets. Volume confirmation ensures breakout validity. Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2147_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    
    # Calculate weekly pivot points from prior week's OHLC
    # Standard formula: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    # We'll use R3/S3 as breakout/fade levels
    
    # Need to align weekly data to daily - we'll calculate weekly pivot for each day
    # using the prior week's complete data
    n_1d = len(close_1d)
    weekly_pivot = np.full(n_1d, np.nan)
    weekly_r3 = np.full(n_1d, np.nan)  # Resistance 3
    weekly_s3 = np.full(n_1d, np.nan)  # Support 3
    
    # Calculate for each day (starting from index 5 to ensure we have prior week)
    for i in range(5, n_1d):
        # Prior week: 5 trading days ago to yesterday
        week_start = max(0, i - 5)
        week_end = i - 1  # yesterday
        
        if week_end >= week_start:
            # Get prior week's OHLC
            week_high = np.max(high_1d[week_start:week_end+1])
            week_low = np.min(low_1d[week_start:week_end+1])
            week_close = close_1d[week_end]
            
            # Calculate pivot
            pivot = (week_high + week_low + week_close) / 3.0
            
            # Calculate R3 and S3
            r3 = week_high + 2.0 * (pivot - week_low)
            s3 = week_low - 2.0 * (week_high - pivot)
            
            weekly_pivot[i] = pivot
            weekly_r3[i] = r3
            weekly_s3[i] = s3
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    
    # Determine bias: 1 if price above pivot (bullish bias), -1 if below (bearish bias)
    # We'll use the pivot level for bias, but R3/S3 for actual breakout/fade levels
    bias_aligned = np.where(close_1d > weekly_pivot, 1, -1)  # This needs to be aligned too
    bias_aligned = align_htf_to_ltf(prices, df_1d, bias_aligned)
    
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
            np.isnan(bias_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price touches or goes below S3 (fade at support)
                if price <= s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches or goes above R3 (take profit at resistance)
                elif price >= r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price touches or goes above R3 (fade at resistance)
                if price >= r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches or goes below S3 (take profit at support)
                elif price <= s3_aligned[i]:
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
            # Long entry: price breaks above R3 AND bullish bias (price above weekly pivot)
            if bias_aligned[i] > 0 and price > r3_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below S3 AND bearish bias (price below weekly pivot)
            elif bias_aligned[i] < 0 and price < s3_aligned[i]:
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