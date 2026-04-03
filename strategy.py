#!/usr/bin/env python3
"""
Experiment #1931: 6h Donchian Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: Weekly pivot levels (calculated from prior 1d data) provide strong institutional support/resistance. 
Strategy: 
- Use 1d data to calculate weekly pivot points (using last 5 trading days)
- Enter on 6h breakout of Donchian(20) channels only when aligned with weekly pivot bias
- Require volume > 1.3x 20-period average for confirmation
- Exit when price returns to weekly pivot point or touches opposite Donchian band
- Works in bull/bear by following weekly institutional flow. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1931_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot points and Donchian channels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from prior 5 trading days (1 week)
    # Using rolling window of 5 days on 1d data
    if len(high_1d) >= 5:
        # Weekly high = max of last 5 daily highs
        weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
        # Weekly low = min of last 5 daily lows
        weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
        # Weekly close = last daily close (close of Friday)
        weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).apply(lambda x: x[-1] if len(x)==5 else np.nan, raw=True).values
        # Weekly pivot = (Weekly High + Weekly Low + Weekly Close) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        # Weekly range = Weekly High - Weekly Low
        weekly_range = weekly_high - weekly_low
        # Weekly R1 = (2 * Weekly Pivot) - Weekly Low
        weekly_r1 = 2 * weekly_pivot - weekly_low
        # Weekly S1 = (2 * Weekly Pivot) - Weekly High
        weekly_s1 = 2 * weekly_pivot - weekly_high
    else:
        # Not enough data for weekly calculation
        weekly_pivot = np.full_like(close_1d, np.nan)
        weekly_r1 = np.full_like(close_1d, np.nan)
        weekly_s1 = np.full_like(close_1d, np.nan)
    
    # Align weekly levels to 6h timeframe (shifted by 1 for completed bars only)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
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
    bars_since_entry = 0
    
    warmup = 50  # sufficient for weekly calculation (5 days) and Donchian(20)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price returns to weekly pivot (mean reversion)
                if price <= weekly_pivot_aligned[i]:
                    exit_signal = True
                # Exit if price touches weekly S1 (strong support)
                elif price <= weekly_s1_aligned[i]:
                    exit_signal = True
            else:  # Short position
                # Exit if price returns to weekly pivot (mean reversion)
                if price >= weekly_pivot_aligned[i]:
                    exit_signal = True
                # Exit if price touches weekly R1 (strong resistance)
                elif price >= weekly_r1_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.3x average)
        volume_spike = vol_ratio[i] > 1.3
        
        if volume_spike:
            # Long entry: price breaks above Donchian high AND price above weekly pivot (bullish bias)
            if price > donchian_high[i] and price > weekly_pivot_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low AND price below weekly pivot (bearish bias)
            elif price < donchian_low[i] and price < weekly_pivot_aligned[i]:
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