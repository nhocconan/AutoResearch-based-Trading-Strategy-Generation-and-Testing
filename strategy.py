#!/usr/bin/env python3
"""
Experiment #2551: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: Donchian breakouts on 6h timeframe aligned with weekly pivot bias (from 1d HTF) 
and volume spikes capture institutional participation during trend acceleration. Weekly pivot 
provides structural support/resistance from higher timeframe, reducing false breakouts. 
Works in bull markets (breakouts with volume) and bear markets (breakdowns with volume). 
Uses discrete position sizing (0.25) to limit fee drag and ensure statistical significance 
with 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2551_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    
    # Calculate weekly pivot points from prior week's 1d data
    # Using simplified weekly: take last 5 trading days (approximation)
    # In practice, we use the most recent completed week's OHLC
    # For each 1d bar, we calculate pivot based on prior 5-day period
    pivot = np.full_like(close_1d, np.nan)
    r1 = np.full_like(close_1d, np.nan)
    s1 = np.full_like(close_1d, np.nan)
    r2 = np.full_like(close_1d, np.nan)
    s2 = np.full_like(close_1d, np.nan)
    r3 = np.full_like(close_1d, np.nan)
    s3 = np.full_like(close_1d, np.nan)
    r4 = np.full_like(close_1d, np.nan)
    s4 = np.full_like(close_1d, np.nan)
    
    lookback = 5  # 5 days ≈ 1 week
    for i in range(lookback, len(close_1d)):
        # Prior week's OHLC (5-day period ending at i-1)
        ph = np.max(high_1d[i-lookback:i])  # prior week high
        pl = np.min(low_1d[i-lookback:i])   # prior week low
        pc = close_1d[i-1]                  prior week close
        
        # Standard pivot calculation
        p = (ph + pl + pc) / 3.0
        pivot[i] = p
        r1[i] = 2 * p - pl
        s1[i] = 2 * p - ph
        r2[i] = p + (ph - pl)
        s2[i] = p - (ph - pl)
        r3[i] = ph + 2 * (p - pl)
        s3[i] = pl - 2 * (ph - p)
        r4[i] = r3[i] + (r2[i] - r1[i])
        s4[i] = s3[i] - (s2[i] - s1[i])
    
    # Determine pivot bias: price above/below weekly pivot
    pivot_bias = np.where(close_1d > pivot, 1, -1)
    pivot_bias_aligned = align_htf_to_ltf(prices, df_1d, pivot_bias)
    
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
        if (np.isnan(pivot_bias_aligned[i]) or
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
        # Require weekly pivot bias for directional filter
        bias = pivot_bias_aligned[i]
        
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike and bias != 0:
            # Long entry: price breaks above Donchian high with bullish bias
            if bias > 0 and price > highest_20[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low with bearish bias
            elif bias < 0 and price < lowest_20[i]:
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