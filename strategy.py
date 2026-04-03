#!/usr/bin/env python3
"""
Experiment #871: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: Donchian breakouts on 6h capture momentum, filtered by 1d weekly pivot bias (above/below weekly pivot) and volume spike (>1.5x average). 
Weekly pivot provides structural bias: price above weekly pivot = bullish bias (favor longs), below = bearish bias (favor shorts). 
This avoids whipsaws in ranging markets by requiring alignment with weekly pivot level. 
Uses discrete position sizing (0.25). Target: 75-150 total trades over 4 years (19-37/year).
Works in bull/bear markets: in bull trends, weekly pivot bias filters for longs; in bear trends, filters for shorts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_871_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from prior week (H+L+C)/3
    # Need at least 5 days for weekly pivot calculation
    weekly_pivot = np.full(len(close_1d), np.nan)
    for i in range(4, len(close_1d)):
        # Prior week: 5 trading days ago to 1 day ago
        week_high = np.max(high_1d[i-5:i])  # exclude current day
        week_low = np.min(low_1d[i-5:i])
        week_close = close_1d[i-1]  # prior day close
        weekly_pivot[i] = (week_high + week_low + week_close) / 3.0
    
    # Pivot bias: 1 = price above weekly pivot (bullish), -1 = below (bearish), 0 = at pivot
    pivot_bias_1d = np.zeros_like(weekly_pivot)
    valid = ~np.isnan(weekly_pivot)
    pivot_bias_1d[valid] = np.where(close_1d[valid] > weekly_pivot[valid], 1,
                                     np.where(close_1d[valid] < weekly_pivot[valid], -1, 0))
    # Align to 6h timeframe
    pivot_bias_6h_aligned = align_htf_to_ltf(prices, df_1d, pivot_bias_1d)
    
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
    
    warmup = max(20, 20, 5)  # sufficient for Donchian, volume MA, weekly pivot
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(pivot_bias_6h_aligned[i]) or
            np.isnan(atr[i])):
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
            
            # Optional: time-based exit after 4 bars (~1d on 6h) to avoid overtrading
            if bars_since_entry > 4:
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
            # Long: price breaks above Donchian upper AND weekly pivot bias bullish
            if price > upper_20[i] and pivot_bias_6h_aligned[i] > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below Donchian lower AND weekly pivot bias bearish
            elif price < lower_20[i] and pivot_bias_6h_aligned[i] < 0:
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