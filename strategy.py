#!/usr/bin/env python3
"""
Experiment #2291: 6h Donchian(20) breakout + 1d weekly pivot + volume confirmation
HYPOTHESIS: Combining 6h Donchian breakouts with 1d weekly pivot levels creates a structure-based edge.
- Primary: 6h Donchian(20) breakout with volume > 2.0x 20-bar average (very strict to limit trades)
- HTF: 1d weekly pivot levels (only long above weekly pivot, short below)
- Exit: ATR(14) trailing stop (2.5*ATR) or opposite pivot level touch
- Target: 50-150 total trades over 4 years (12-37/year) - optimized for 6h timeframe
- Works in bull markets (breakout continuation) and bear markets (mean reversion at pivots)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2291_6h_donchian20_1d_pivot_vol_v1"
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
    
    # Calculate weekly pivot from prior week's 1d data
    # Need to group by week - we'll approximate using rolling 5-day (1 week trading)
    # Weekly high = max of prior 5 days, weekly low = min of prior 5 days, weekly close = close 5 days ago
    if len(high_1d) >= 5:
        weekly_high = np.concatenate([np.full(4, np.nan), pd.Series(high_1d).rolling(window=5, min_periods=5).max().values[4:]])
        weekly_low = np.concatenate([np.full(4, np.nan), pd.Series(low_1d).rolling(window=5, min_periods=5).min().values[4:]])
        weekly_close = np.concatenate([np.full(4, np.nan), close_1d[4:]])
    else:
        weekly_high = np.full_like(high_1d, np.nan)
        weekly_low = np.full_like(low_1d, np.nan)
        weekly_close = np.full_like(close_1d, np.nan)
    
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Weekly support/resistance levels
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    
    # === 6h Indicators: Donchian(20), Volume MA(20), ATR(14) ===
    # Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
    # Volume MA for spike detection (very strict threshold to limit trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
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
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches weekly S1 (mean reversion)
                elif price <= s1_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches weekly R1 (mean reversion)
                elif price >= r1_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 2.0x average - very strict to limit trades)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: price breaks above upper Donchian AND above weekly pivot
            if price > donchian_upper[i] and price > pivot_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND below weekly pivot
            elif price < donchian_lower[i] and price < pivot_aligned[i]:
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