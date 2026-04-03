#!/usr/bin/env python3
"""
Experiment #355: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: Donchian channel breakouts on 6h timeframe, filtered by weekly pivot direction (price above/below weekly pivot) 
and 1d volume spike confirmation, creates a robust trend-following strategy that works in both bull and bear markets. 
The weekly pivot provides institutional reference points, Donchian breakouts capture momentum, and volume confirms 
institutional participation. Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_breakout_weekly_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly pivot (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot point (typical price: (H+L+C)/3) from prior completed week
    weekly_pivot = np.full(n, np.nan)
    if len(df_1w) >= 1:
        typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, typical_price.values)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume spike confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= donchian_period:
        # Calculate rolling max/min for Donchian channels
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_high = high_series.rolling(window=donchian_period, min_periods=donchian_period).max().values
        donchian_low = low_series.rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(100, donchian_period)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Price relative to weekly pivot ---
        price_above_weekly_pivot = close[i] > weekly_pivot_aligned[i]
        price_below_weekly_pivot = close[i] < weekly_pivot_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Donchian band
                if close[i] <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Donchian band
                if close[i] >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Break above Donchian high with volume and price above weekly pivot
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            price_above_weekly_pivot
        )
        
        # Short: Break below Donchian low with volume and price below weekly pivot
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            price_below_weekly_pivot
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals