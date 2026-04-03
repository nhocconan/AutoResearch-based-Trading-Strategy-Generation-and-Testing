#!/usr/bin/env python3
"""
Experiment #1079: 6h Donchian(20) Breakout + 12h/1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: Donchian(20) breakouts capture institutional order flow. Weekly pivots (from 1d data) define the major trend direction. Trade only in direction of weekly pivot (long above weekly pivot, short below). Volume confirmation (>1.5x average) filters false breakouts. Uses discrete position sizing (0.25) to manage drawdown. Target: 75-150 total trades over 4 years (19-37/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1079_6h_donchian20_12h_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for weekly pivot calculation (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate weekly pivot points from prior week (using 12h data ~2 weeks)
    # Need ~14 bars of 12h data for 1 week (14*12h = 168h = 1 week)
    lookback = 14
    pivot_high = np.full(n, np.nan)
    pivot_low = np.full(n, np.nan)
    pivot_point = np.full(n, np.nan)
    
    for i in range(lookback, len(high_12h)):
        # Prior week high/low/close
        week_high = np.max(high_12h[i-lookback:i])
        week_low = np.min(low_12h[i-lookback:i])
        week_close = close_12h[i-1]  # prior bar close
        
        # Standard pivot calculation
        pp = (week_high + week_low + week_close) / 3.0
        pivot_point[i] = pp
        pivot_high[i] = 2 * pp - week_low  # R1
        pivot_low[i] = 2 * pp - week_high   # S1
    
    # Align to 6h timeframe
    pivot_point_6h = align_htf_to_ltf(prices, df_12h, pivot_point)
    pivot_high_6h = align_htf_to_ltf(prices, df_12h, pivot_high)
    pivot_low_6h = align_htf_to_ltf(prices, df_12h, pivot_low)
    
    # === 6h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = 20  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(pivot_point_6h[i]) or np.isnan(pivot_high_6h[i]) or np.isnan(pivot_low_6h[i])):
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
            
            # Optional: time-based exit after 12 bars (~3d on 6h) to avoid overtrading
            if bars_since_entry > 12:
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
            # Weekly pivot direction filter: price above/below weekly pivot
            price_above_pivot = price > pivot_point_6h[i]
            price_below_pivot = price < pivot_point_6h[i]
            
            # Breakout: price breaks above Donchian high OR below Donchian low
            if price > donch_high[i] and price_above_pivot:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < donch_low[i] and price_below_pivot:
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

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA for half period
    wma_half = np.zeros_like(close)
    for i in range(half_period, len(close)):
        weights = np.arange(1, half_period + 1)
        wma_half[i] = np.dot(close[i - half_period + 1:i + 1], weights) / weights.sum()
    
    # WMA for full period
    wma_full = np.zeros_like(close)
    for i in range(period, len(close)):
        weights = np.arange(1, period + 1)
        wma_full[i] = np.dot(close[i - period + 1:i + 1], weights) / weights.sum()
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA of raw HMA with sqrt_period
    hma = np.zeros_like(close)
    for i in range(sqrt_period, len(close)):
        weights = np.arange(1, sqrt_period + 1)
        hma[i] = np.dot(raw_hma[i - sqrt_period + 1:i + 1], weights) / weights.sum()
    
    return hma