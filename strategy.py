#!/usr/bin/env python3
"""
Experiment #151: 6h Donchian(20) breakout + 1d Weekly Pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot levels (R3/S3 for fade, R4/S4 for breakout) and volume confirmation capture institutional participation in both bull and bear markets. Weekly pivots provide structural support/resistance from higher timeframe (1w). Volume confirmation (>1.5x average) ensures breakout validity. Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_151_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot points (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # Calculate weekly pivot points from prior week's 1d OHLC
    # Using prior week's high/low/close (shifted by 5 trading days)
    prior_week_high = high_1d.rolling(window=5, min_periods=5).max().shift(5).values
    prior_week_low = low_1d.rolling(window=5, min_periods=5).min().shift(5).values
    prior_week_close = close_1d.rolling(window=5, min_periods=5).last().shift(5).values
    
    # Weekly pivot calculation
    pp = (prior_week_high + prior_week_low + prior_week_close) / 3.0
    r1 = 2 * pp - prior_week_low
    s1 = 2 * pp - prior_week_high
    r2 = pp + (prior_week_high - prior_week_low)
    s2 = pp - (prior_week_high - prior_week_low)
    r3 = r2 + (prior_week_high - prior_week_low)
    s3 = s2 - (prior_week_high - prior_week_low)
    r4 = r3 + (prior_week_high - prior_week_low)
    s4 = s3 - (prior_week_high - prior_week_low)
    
    # Align weekly pivot levels to 6h timeframe (shifted by 1 for completed week only)
    pp_6h = align_htf_to_ltf(prices, df_1d, pp)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(pp_6h[i]) or np.isnan(r3_6h[i]) or
            np.isnan(s3_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Weekly Pivot Logic ---
        # Fade at R3/S3 (price rejects extreme levels)
        fade_up = price < r3_6h[i] and price > s3_6h[i] and abs(price - r3_6h[i]) < (r4_6h[i] - r3_6h[i]) * 0.2
        fade_down = price > s3_6h[i] and price < r3_6h[i] and abs(price - s3_6h[i]) < (s3_6h[i] - s4_6h[i]) * 0.2
        
        # Breakout continuation at R4/S4 (price breaks extreme levels with momentum)
        breakout_continuation_up = price > r4_6h[i]
        breakout_continuation_down = price < s4_6h[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 6 bars (~36h on 6h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Donchian breakout OR fade at S3/S4 OR breakout continuation above R4
            if (breakout_up or fade_down or breakout_continuation_up) and price > pp_6h[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Donchian breakout OR fade at R3/R4 OR breakout continuation below S4
            elif (breakout_down or fade_up or breakout_continuation_down) and price < pp_6h[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals