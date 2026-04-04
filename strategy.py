#!/usr/bin/env python3
"""
Experiment #3887: 6h Donchian(20) breakout + 1d weekly pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d weekly pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) capture medium-term momentum with reduced whipsaw in both bull and bear markets. Volume > 1.5x MA(30) confirms participation. ATR(14) trailing stop (2.0x) manages risk. Target: 75-150 trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3887_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot points (using last completed week) ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate weekly pivot from prior week's daily OHLC (requires 5 daily bars)
    # We'll use the prior week's high, low, close to calculate pivot for current week
    # For simplicity, we calculate daily pivots and use the most recent completed day's pivot
    # But to align with weekly concept, we'll use prior 5-day period's extreme values
    lookback_week = 5
    if len(df_1d) >= lookback_week:
        week_high = pd.Series(df_1d['high'].values).rolling(window=lookback_week, min_periods=lookback_week).max().values
        week_low = pd.Series(df_1d['low'].values).rolling(window=lookback_week, min_periods=lookback_week).min().values
        week_close = pd.Series(df_1d['close'].values).rolling(window=lookback_week, min_periods=lookback_week).last().values
        # Weekly pivot point: (week_high + week_low + week_close) / 3
        weekly_pivot = (week_high + week_low + week_close) / 3.0
        # Weekly range
        weekly_range = week_high - week_low
        # Camarilla-style weekly levels
        r3 = weekly_pivot + weekly_range * 1.1 / 2
        s3 = weekly_pivot - weekly_range * 1.1 / 2
        r4 = weekly_pivot + weekly_range * 1.1
        s4 = weekly_pivot - weekly_range * 1.1
        # Align to LTF
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        # Not enough data for weekly calculation
        weekly_pivot_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(30) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[30:] = volume[30:] / vol_ma[30:]
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 30, lookback_week)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian lower band (trend reversal)
                elif price < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian upper band (trend reversal)
                elif price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) to filter noise
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Determine pivot-based bias
            # Near S3/R3: mean reversion bias
            # Beyond S4/R4: breakout continuation bias
            near_s3 = abs(price - s3_aligned[i]) < (r3_aligned[i] - s3_aligned[i]) * 0.1
            near_r3 = abs(price - r3_aligned[i]) < (r3_aligned[i] - s3_aligned[i]) * 0.1
            beyond_r4 = price > r4_aligned[i]
            beyond_s4 = price < s4_aligned[i]
            
            # Long conditions
            long_mean_revert = near_s3 and price > s3_aligned[i]  # Bounce off S3
            long_breakout = beyond_r4 and price > highest_high[i-1]  # Break above R4 with momentum
            
            # Short conditions
            short_mean_revert = near_r3 and price < r3_aligned[i]  # Rejection at R3
            short_breakout = beyond_s4 and price < lowest_low[i-1]  # Break below S4 with momentum
            
            if (long_mean_revert or long_breakout) and not (short_mean_revert or short_breakout):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif (short_mean_revert or short_breakout) and not (long_mean_revert or long_breakout):
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