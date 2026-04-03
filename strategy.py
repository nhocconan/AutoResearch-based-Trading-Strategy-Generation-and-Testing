#!/usr/bin/env python3
"""
Experiment #1947: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: Weekly pivot levels provide institutional support/resistance from higher timeframe structure. 
Donchian(20) breakouts on 6h timeframe capture momentum when aligned with weekly bias and confirmed by volume spikes.
Works in bull/bear markets by following weekly institutional flow. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1947_6h_donchian20_1w_pivot_vol_v1"
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
    
    # Calculate weekly pivot from 1d data (aggregate to weekly)
    # We'll compute weekly pivot using the last 5 days of 1d data
    # Weekly Pivot = (Weekly High + Weekly Low + Weekly Close) / 3
    # For efficiency, we approximate using rolling window on 1d data
    # In practice, we use the most recent completed weekly bar
    
    # Calculate 5-day rolling high/low/close for weekly approximation
    # Using 5 days = 1 week (approximation for alignment)
    high_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    low_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    close_5d = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly Pivot levels
    weekly_pivot = (high_5d + low_5d + close_5d) / 3.0
    weekly_range = high_5d - low_5d
    
    # Weekly support/resistance levels (similar to Camarilla but simpler)
    # R2 = Pivot + Range
    # S2 = Pivot - Range
    r2_1d = weekly_pivot + weekly_range
    s2_1d = weekly_pivot - weekly_range
    
    # Weekly bias: price above/below weekly pivot
    weekly_bias = np.where(close_5d > weekly_pivot, 1, -1)
    
    # Align weekly levels to 6h timeframe (shifted by 1 for completed bars only)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1d, weekly_bias)
    
    # === 6h Indicators: Donchian(20) for breakout detection ===
    # Donchian channels: upper = max(high, lookback), lower = min(low, lookback)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
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
    
    warmup = 50  # sufficient for Donchian(20), weekly calculations, and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(r2_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or
            np.isnan(weekly_bias_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions: mean reversion to weekly pivot or opposite Donchian touch
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price returns to weekly pivot (mean reversion)
                if price <= weekly_pivot_aligned[i]:
                    exit_signal = True
                # Exit if price touches Donchian lower band (failed breakout)
                elif price <= donchian_low[i]:
                    exit_signal = True
            else:  # Short position
                # Exit if price returns to weekly pivot (mean reversion)
                if price >= weekly_pivot_aligned[i]:
                    exit_signal = True
                # Exit if price touches Donchian upper band (failed breakdown)
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
        # Require weekly bias alignment for directional filter
        weekly_bias_val = weekly_bias_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND weekly bullish bias
            if weekly_bias_val > 0 and price > donchian_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND weekly bearish bias
            elif weekly_bias_val < 0 and price < donchian_low[i]:
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