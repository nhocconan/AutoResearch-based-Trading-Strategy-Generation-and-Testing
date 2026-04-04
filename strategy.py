#!/usr/bin/env python3
"""
Experiment #2635: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: Weekly pivot points provide institutional reference levels. 
Donchian breakouts aligned with weekly trend and volume spikes capture 
institutional participation in 6h timeframe. Uses 1w for pivot/direction, 
6h only for entry timing. Target: 75-150 total trades over 4 years.
Works in bull (breakout continuation) and bear (mean reversion at extremes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2635_6h_donchian20_weekly_pivot_vol_v1"
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
    n_1d = len(close_1d)
    
    # === Calculate Weekly Pivot Points from Daily OHLC ===
    # Standard formula: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    # R4 = H + 3*(P - L), S4 = L - 3*(H - P)
    
    # Calculate daily pivot components
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    hl_range_1d = high_1d - low_1d
    
    # Weekly aggregation: use last 5 days (approximation)
    # We'll use rolling window of 5 days for weekly levels
    window = 5
    if n_1d >= window:
        # Weekly high/low/close from daily data
        weekly_high = pd.Series(high_1d).rolling(window=window, min_periods=window).max().values
        weekly_low = pd.Series(low_1d).rolling(window=window, min_periods=window).min().values
        weekly_close = pd.Series(close_1d).rolling(window=window, min_periods=window).last().values
        
        # Weekly pivot points
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_range = weekly_high - weekly_low
        
        # Weekly support/resistance levels
        weekly_r1 = 2 * weekly_pivot - weekly_low
        weekly_s1 = 2 * weekly_pivot - weekly_high
        weekly_r2 = weekly_pivot + weekly_range
        weekly_s2 = weekly_pivot - weekly_range
        weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
        weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
        weekly_r4 = weekly_high + 3 * (weekly_range)
        weekly_s4 = weekly_low - 3 * (weekly_range)
        
        # Determine weekly trend: price above/below weekly pivot
        weekly_trend = np.where(weekly_close > weekly_pivot, 1, -1)
        
        # Align to 6h timeframe
        pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
        r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
        s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
        r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
        s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
        r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, weekly_r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, weekly_s4)
        trend_aligned = align_htf_to_ltf(prices, df_1d, weekly_trend)
    else:
        # Not enough data - return zeros
        pivot_aligned = r1_aligned = s1_aligned = r2_aligned = s2_aligned = np.full(n, np.nan)
        r3_aligned = s3_aligned = r4_aligned = s4_aligned = trend_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pivot_aligned[i]) or np.isnan(trend_aligned[i]) or
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
        # Weekly trend filter
        weekly_trend = trend_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Long entry: price breaks above Donchian high with bullish weekly trend
            if weekly_trend > 0 and price > highest_20[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low with bearish weekly trend
            elif weekly_trend < 0 and price < lowest_20[i]:
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