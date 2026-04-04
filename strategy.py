#!/usr/bin/env python3
"""
Experiment #3152: 12h Camarilla Pivot + 1w Volume Spike + Chop Regime Filter
HYPOTHESIS: 12h strategies using Camarilla pivot levels from 1d timeframe capture institutional reaction points with low trade frequency.
Weekly volume spike (>2.5x average) confirms breakout significance.
Choppiness Index regime filter (CHOP > 61.8 = range) enables mean reversion at pivot levels.
Position size 0.25. Target: 75-150 total trades over 4 years (19-37/year).
Designed to work in bull markets (breakout continuation) and bear markets (mean reversion at extremes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3152_12h_camarilla1w_vol_chop_v1"
timeframe = "12h"
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
    # Camarilla: R4 = close + 1.5*(high-low)*1.1/2, R3 = close + 1.25*(high-low)*1.1/2, etc.
    # We use R3, S3 as primary levels
    rng_1d = high_1d - low_1d
    camarilla_r3_1d = close_1d + 1.25 * rng_1d * 1.1 / 2
    camarilla_s3_1d = close_1d - 1.25 * rng_1d * 1.1 / 2
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # === HTF: 1w data for volume average (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # === 12h Indicators: Choppiness Index (14) for regime detection ===
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        """Calculate Choppiness Index: higher = more choppy/ranging"""
        tr1 = high_arr[1:] - low_arr[1:]
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        atr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum()
        highest_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max()
        lowest_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min()
        
        chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(window)
        return chop.values
    
    chop = calculate_chop(high, low, close, 14)
    
    # === 12h Indicators: Volume ratio vs weekly average ===
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma_1w_aligned[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(50, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit conditions: price reaches opposite pivot level or chop regime changes
            if position_side > 0:  # Long
                # Exit if price reaches S3 (mean reversion target)
                if price <= camarilla_s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if chop drops below 38.2 (trending regime) - trail with price action
                elif chop[i] < 38.2 and price < close[i-1]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Exit if price reaches R3 (mean reversion target)
                if price >= camarilla_r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if chop drops below 38.2 (trending regime) - trail with price action
                elif chop[i] < 38.2 and price > close[i-1]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.5x weekly average) for confirmation
        volume_spike = vol_ratio[i] > 2.5
        
        # Require choppy/ranging regime (CHOP > 61.8) for mean reversion setup
        choppy_regime = chop[i] > 61.8
        
        if volume_spike and choppy_regime:
            # Long entry: price touches S3 level with bullish rejection (close > open)
            if low[i] <= camarilla_s3_aligned[i] and close[i] > open[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: price touches R3 level with bearish rejection (close < open)
            elif high[i] >= camarilla_r3_aligned[i] and close[i] < open[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals