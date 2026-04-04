#!/usr/bin/env python3
"""
Experiment #2667: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d weekly pivot levels (R3/S3 for fade, R4/S4 for continuation) 
and volume spikes capture institutional participation. Weekly pivots provide structural support/resistance 
that works in both bull and bear markets. Uses 1d for pivot levels and trend bias, 6h only for entry timing. 
Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2667_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivots and trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # We'll use prior daily bar as proxy for weekly calculation (simplified)
    # For proper weekly, we'd need to group by week, but prior day OHLC gives reasonable pivot levels
    pivot_1d = (high_1d[:-2] + low_1d[:-2] + close_1d[:-2]) / 3.0  # prior bar
    r1_1d = 2 * pivot_1d - low_1d[:-2]
    s1_1d = 2 * pivot_1d - high_1d[:-2]
    r2_1d = pivot_1d + (high_1d[:-2] - low_1d[:-2])
    s2_1d = pivot_1d - (high_1d[:-2] - low_1d[:-2])
    r3_1d = high_1d[:-2] + 2 * (pivot_1d - low_1d[:-2])
    s3_1d = low_1d[:-2] - 2 * (high_1d[:-2] - pivot_1d)
    r4_1d = r3_1d + (high_1d[:-2] - low_1d[:-2])
    s4_1d = s3_1d - (high_1d[:-2] - low_1d[:-2])
    
    # Align pivot levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 1d EMA(50) for trend bias
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
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
        if (np.isnan(trend_1d_aligned[i]) or
            np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
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
        # Require 1d trend alignment for bias filter
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Determine pivot-based entry logic
            pivot = pivot_1d_aligned[i]
            r3 = r3_1d_aligned[i]
            s3 = s3_1d_aligned[i]
            r4 = r4_1d_aligned[i]
            s4 = s4_1d_aligned[i]
            
            # Long entry conditions:
            # 1. Continuation: price breaks above R4 with uptrend on 1d
            # 2. Fade: price rejects from S3 with uptrend on 1d (mean reversion)
            if trend_bias > 0:
                if (price > r4 and price > highest_20[i]) or \
                   (price < s3 and price > low[i] and close[i] > open[i]):  # bullish rejection at S3
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = SIZE
            # Short entry conditions:
            # 1. Continuation: price breaks below S4 with downtrend on 1d
            # 2. Fade: price rejects from R3 with downtrend on 1d (mean reversion)
            elif trend_bias < 0:
                if (price < s4 and price < lowest_20[i]) or \
                   (price > r3 and price < high[i] and close[i] < open[i]):  # bearish rejection at R3
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