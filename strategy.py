#!/usr/bin/env python3
"""
Experiment #779: 6h Donchian(20) Breakout + 12h Volume Spike + Weekly Pivot Filter
HYPOTHESIS: Donchian breakouts on 6h capture momentum, filtered by 12h volume confirmation (>2.0x average) 
and weekly pivot direction (long above weekly pivot, short below). Weekly pivot acts as regime filter: 
in bull markets, price tends to stay above weekly pivot; in bear markets, below. This avoids counter-trend 
breakouts. Uses discrete position sizing (0.25). Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_779_6h_donchian20_12h_vol_weeklypivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume spike filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # Calculate Volume MA(20) on 12h for spike detection
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = np.ones(len(volume_12h))
    vol_ratio_12h[20:] = volume_12h[20:] / vol_ma_12h[20:]
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    # We need to shift by 1 week to avoid look-ahead
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(5)  # 5 trading days = 1 week
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(5)
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(5)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def donchian_channel(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channel(high, low, 20)
    
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
    
    warmup = max(20, 20, 20)  # sufficient for Donchian, volume MA, weekly pivot
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or
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
        # Volume confirmation: require volume spike (> 2.0x average on 12h)
        volume_spike = vol_ratio_12h_aligned[i] > 2.0
        
        if volume_spike:
            # Long: price breaks above Donchian upper AND price above weekly pivot
            if price > upper_20[i] and price > weekly_pivot_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below Donchian lower AND price below weekly pivot
            elif price < lower_20[i] and price < weekly_pivot_aligned[i]:
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