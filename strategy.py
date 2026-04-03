#!/usr/bin/env python3
"""
Experiment #1971: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: Weekly pivot levels from 1d timeframe provide institutional support/resistance that align with 6h Donchian breakouts. 
Strategy: 
- Calculate weekly pivot (using 1d OHLC aggregated to weekly) 
- Determine bias: price above weekly pivot = bullish, below = bearish
- Enter on 6h breakout of Donchian(20) high/low only when aligned with weekly bias and volume > 1.5x 20-period average
- Exit when price returns to weekly pivot (mean reversion) or opposite Donchian level is touched
- Works in bull/bear markets by following weekly institutional flow. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1971_6h_donchian20_1w_pivot_vol_v1"
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
    
    # Calculate weekly pivot from daily data (simulate weekly aggregation)
    # Weekly High = max of last 5 daily highs (approximation for 5 trading days)
    # Weekly Low = min of last 5 daily lows
    # Weekly Close = last daily close
    weekly_high = np.full(n_1d, np.nan)
    weekly_low = np.full(n_1d, np.nan)
    weekly_close = close_1d.copy()
    
    for i in range(4, n_1d):  # Need 5 days for weekly
        weekly_high[i] = np.max(high_1d[i-4:i+1])
        weekly_low[i] = np.min(low_1d[i-4:i+1])
    
    # Weekly pivot = (Weekly High + Weekly Low + Weekly Close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe (shifted by 1 for completed bars only)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # === 6h Indicators: Donchian(20) channels ===
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
    
    warmup = 50  # sufficient for Donchian(20) and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ratio[i])):
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
                # Exit if price touches Donchian low (strong support)
                elif price <= donchian_low[i]:
                    exit_signal = True
            else:  # Short position
                # Exit if price returns to weekly pivot (mean reversion)
                if price >= weekly_pivot_aligned[i]:
                    exit_signal = True
                # Exit if price touches Donchian high (strong resistance)
                elif price >= donchian_high[i]:
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
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
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