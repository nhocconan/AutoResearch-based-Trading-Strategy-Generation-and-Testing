#!/usr/bin/env python3
"""
Experiment #911: 6h Donchian(20) + 1d Weekly Pivot Direction + Volume Spike
HYPOTHESIS: Donchian breakouts on 6h capture momentum, filtered by 1d weekly pivot direction 
(price above/below weekly pivot) and volume confirmation (>1.5x average). Long when price 
breaks above Donchian upper AND price > weekly pivot AND volume spike. Short when price 
breaks below Donchian lower AND price < weekly pivot AND volume spike. Uses discrete 
position sizing (0.25) to balance risk and reward. Target: 75-150 total trades over 4 years 
(19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_911_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot points from prior week (using 1d OHLC)
    # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    # We need to shift by 5 days to get prior week's data (assuming 5 trading days)
    # But since we have daily data, we can use rolling window of 5 and shift by 5
    # However, for simplicity and to avoid look-ahead, we'll use prior day's data to approximate
    # Better approach: use weekly resampled data, but we must use get_htf_data for 1w
    # Let's get 1w data for proper weekly pivot
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) > 0:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # Weekly pivot = (High + Low + Close) / 3
        weekly_pivot_1w = (high_1w + low_1w + close_1w) / 3.0
        
        # Align weekly pivot to 6h timeframe
        weekly_pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_1w)
    else:
        # Fallback to daily approximation if 1w fails
        weekly_pivot_1w_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def donchian_channel(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channel(high, low, 20)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 20)  # sufficient for Donchian, volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            (len(df_1w) > 0 and np.isnan(weekly_pivot_1w_aligned[i]))):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
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
            
            # Optional: time-based exit after 8 bars (~48h on 6h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Get weekly pivot value (handle case where 1w data might not be available)
            if len(df_1w) > 0:
                weekly_pivot_val = weekly_pivot_1w_aligned[i]
                # Long: price breaks above Donchian upper AND price > weekly pivot
                if price > upper_20[i] and price > weekly_pivot_val:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Short: price breaks below Donchian lower AND price < weekly pivot
                elif price < lower_20[i] and price < weekly_pivot_val:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:
                # Fallback: use price relative to prior day's pivot approximation
                # Prior day typical price = (high + low + close)/3
                prior_typical = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0 if i > 0 else close[i]
                # Long: price breaks above Donchian upper AND price > prior typical
                if price > upper_20[i] and price > prior_typical:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Short: price breaks below Donchian lower AND price < prior typical
                elif price < lower_20[i] and price < prior_typical:
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