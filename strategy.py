#!/usr/bin/env python3
"""
Experiment #167: 6h Donchian Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h Donchian channel breakouts aligned with weekly pivot-derived trend (from 1d HTF) and volume confirmation capture strong directional moves while avoiding false breakouts. Weekly pivot provides structural bias (bull/bear/range) based on prior week's price action, effective in both bull and bear markets. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_167_6h_donchian_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from prior week (using 1d data)
    # Weekly high/low/close from prior completed week
    week_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1)  # prior week
    week_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1)
    week_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(1)
    
    # Weekly pivot and support/resistance levels
    pp = (week_high + week_low + week_close) / 3.0
    r1 = 2 * pp - week_low
    s1 = 2 * pp - week_high
    r2 = pp + (week_high - week_low)
    s2 = pp - (week_high - week_low)
    r3 = week_high + 2 * (pp - week_low)
    s3 = week_low - 2 * (week_high - pp)
    
    # Trend bias: price above weekly R1 = bull, below S1 = bear, between = range
    weekly_bull = df_1d['close'].values > r1
    weekly_bear = df_1d['close'].values < s1
    weekly_range = ~(weekly_bull | weekly_bear)
    
    # Align to 6h timeframe
    weekly_bull_aligned = align_htf_to_ltf(prices, df_1d, weekly_bull)
    weekly_bear_aligned = align_htf_to_ltf(prices, df_1d, weekly_bear)
    weekly_range_aligned = align_htf_to_ltf(prices, df_1d, weekly_range)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period lookbacks and 5-day weekly
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(weekly_bull_aligned[i]) or
            np.isnan(weekly_bear_aligned[i]) or np.isnan(weekly_range_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
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
        # Long: Price breaks above Donchian HIGH, volume spike, weekly bull bias
        if (close[i] > donchian_high[i-1] and  # break above previous period's high
            volume_spike and 
            weekly_bull_aligned[i]):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Price breaks below Donchian LOW, volume spike, weekly bear bias
        elif (close[i] < donchian_low[i-1] and  # break below previous period's low
              volume_spike and 
              weekly_bear_aligned[i]):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals