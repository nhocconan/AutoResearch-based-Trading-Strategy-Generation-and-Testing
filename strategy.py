#!/usr/bin/env python3
"""
Experiment #2351: 6h Donchian(20) breakout + 1d weekly pivot + volume confirmation
HYPOTHESIS: Donchian breakouts aligned with weekly pivot levels (from 1d HTF) capture institutional 
participation at key structural levels. Weekly pivot provides bias (R3/S3 for fade, R4/S4 for breakout), 
volume confirms participation. Works in bull/bear by fading extremes in range and continuing breaks in trend.
Target: 50-150 trades over 4 years (12-37/year). Size: 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2351_6h_donchian20_1d_weeklypivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot from prior week (using last 5 trading days approx)
    # For simplicity: use prior 5-day high/low/close to compute weekly pivot
    if len(close_1d) >= 5:
        # Rolling window of 5 days for weekly approximation
        high_5 = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
        low_5 = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
        close_5 = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values  # last close in window
        
        # Weekly pivot: P = (H + L + C) / 3
        weekly_p = (high_5 + low_5 + close_5) / 3.0
        # Weekly R1, S1
        weekly_r1 = 2 * weekly_p - low_5
        weekly_s1 = 2 * weekly_p - high_5
        # Weekly R2, S2
        weekly_r2 = weekly_p + (high_5 - low_5)
        weekly_s2 = weekly_p - (high_5 - low_5)
        # Weekly R3, S3 (strong support/resistance)
        weekly_r3 = high_5 + 2 * (weekly_p - low_5)
        weekly_s3 = low_5 - 2 * (high_5 - weekly_p)
        # Weekly R4, S4 (extreme breakout levels)
        weekly_r4 = weekly_p + 3 * (high_5 - low_5)
        weekly_s4 = weekly_p - 3 * (high_5 - low_5)
        
        # Align to 6h timeframe (shifted by 1 for completed weekly bar)
        weekly_p_aligned = align_htf_to_ltf(prices, df_1d, weekly_p)
        weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
        weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
        weekly_r4_aligned = align_htf_to_ltf(prices, df_1d, weekly_r4)
        weekly_s4_aligned = align_htf_to_ltf(prices, df_1d, weekly_s4)
    else:
        # Not enough data
        weekly_p_aligned = np.full(n, np.nan)
        weekly_r3_aligned = np.full(n, np.nan)
        weekly_s3_aligned = np.full(n, np.nan)
        weekly_r4_aligned = np.full(n, np.nan)
        weekly_s4_aligned = np.full(n, np.nan)
    
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
        if (np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i]) or
            np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or
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
                # Exit if price drops below weekly S3 (strong support) or Donchian low
                if price < weekly_s3_aligned[i] or price < lowest_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches weekly R4 (take profit at extreme)
                elif price > weekly_r4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises above weekly R3 (strong resistance) or Donchian high
                if price > weekly_r3_aligned[i] or price > highest_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches weekly S4 (take profit at extreme)
                elif price < weekly_s4_aligned[i]:
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
            # Fade at weekly R3/S3 (extreme levels in range)
            # Short at R3 with rejection (price < open after hitting R3)
            # Long at S3 with rejection (price > open after hitting S3)
            if price > weekly_r3_aligned[i] * 0.999 and price < weekly_r3_aligned[i] * 1.001:
                # Tested R3, check for rejection (close below open)
                if close[i] < prices["open"].iloc[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE
            elif price > weekly_s3_aligned[i] * 0.999 and price < weekly_s3_aligned[i] * 1.001:
                # Tested S3, check for rejection (close above open)
                if close[i] > prices["open"].iloc[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = SIZE
            # Breakout continuation at weekly R4/S4
            elif price > weekly_r4_aligned[i]:
                # Break above R4, go long with trend
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif price < weekly_s4_aligned[i]:
                # Break below S4, go short with trend
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