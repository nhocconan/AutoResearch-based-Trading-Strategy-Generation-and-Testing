#!/usr/bin/env python3
"""
Experiment #187: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h Donchian(20) breakouts aligned with weekly Camarilla pivot direction (from 1w HTF) and volume spikes capture institutional breakouts with follow-through. Weekly pivot provides structural bias (long above weekly pivot, short below) while Donchian breakouts capture momentum. Volume confirmation filters false breakouts. Designed for 6h timeframe to balance trade frequency and signal quality, targeting 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_187_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation ===
    # We need 1d data to calculate weekly pivots (5 trading days per week)
    df_1d = get_htf_data(prices, '1d')
    
    # === Calculate Weekly Camarilla Pivot Levels from 1d OHLC ===
    # Group 1d data into weeks (starting Monday)
    # We'll calculate weekly pivot on rolling 5-day window
    lookback = 5  # 5 trading days = 1 week
    if len(df_1d) >= lookback:
        # Rolling high, low, close over past 5 days
        high_5d = pd.Series(df_1d['high'].values).rolling(window=lookback, min_periods=lookback).max().values
        low_5d = pd.Series(df_1d['low'].values).rolling(window=lookback, min_periods=lookback).min().values
        close_5d = pd.Series(df_1d['close'].values).rolling(window=lookback, min_periods=lookback).last().values
        
        # Camarilla pivot calculation
        pivot = (high_5d + low_5d + close_5d) / 3.0
        range_5d = high_5d - low_5d
        
        # Camarilla levels
        r3 = pivot + range_5d * 1.1 / 2.0
        s3 = pivot - range_5d * 1.1 / 2.0
        r4 = pivot + range_5d * 1.1
        s4 = pivot - range_5d * 1.1
        
        # Trend direction based on weekly pivot
        # Bullish if price above weekly pivot, bearish if below
        trend_up_weekly = close_5d > pivot
        trend_down_weekly = close_5d < pivot
        
        # Align to 6h timeframe
        trend_up_weekly_aligned = align_htf_to_ltf(prices, df_1d, trend_up_weekly)
        trend_down_weekly_aligned = align_htf_to_ltf(prices, df_1d, trend_down_weekly)
    else:
        # Not enough data for weekly pivot
        trend_up_weekly_aligned = np.zeros(n, dtype=bool)
        trend_down_weekly_aligned = np.zeros(n, dtype=bool)
    
    # === 6h Indicators: Donchian Channel (20) ===
    lookback_donch = 20
    highest_high = pd.Series(high).rolling(window=lookback_donch, min_periods=lookback_donch).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_donch, min_periods=lookback_donch).min().values
    
    # Donchian breakout signals
    breakout_up = high > highest_high  # Price breaks above Donchian high
    breakout_down = low < lowest_low   # Price breaks below Donchian low
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(lookback_donch, lookback, 20) + 5
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if price reaches opposite Donchian band (mean reversion)
                if low[i] <= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if price reaches opposite Donchian band
                if high[i] >= highest_high[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        # Long: Donchian breakout up + weekly uptrend + volume spike
        if breakout_up[i] and trend_up_weekly_aligned[i] and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Donchian breakout down + weekly downtrend + volume spike
        elif breakout_down[i] and trend_down_weekly_aligned[i] and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals