#!/usr/bin/env python3
"""
Experiment #1547: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot direction from 1d timeframe, with volume confirmation (>1.8x average), capture medium-term swings in both bull and bear markets. Weekly pivot provides structural support/resistance from higher timeframe, reducing false breakouts. Position size fixed at 0.25 to balance return and drawdown. Target: 75-175 total trades over 4 years (19-44/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1547_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot and trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot points from prior week (using 1d data)
    # We need to group by week and calculate pivot for prior week
    # For simplicity, we'll use rolling window of 5 days (1 week) to approximate
    # In practice, we'd use actual weekly grouping, but rolling 5 is close enough for 1d data
    # Pivot = (High + Low + Close) / 3
    # We'll calculate for each day, then use prior week's value
    # To avoid lookahead, we shift by 5 (prior week)
    typical_1d = (high_1d + low_1d + close_1d) / 3.0
    # Weekly pivot: typical price of prior week (5-day shift)
    weekly_pivot = pd.Series(typical_1d).rolling(window=5, min_periods=5).mean().shift(5).values
    # Support 1: 2*Pivot - High
    weekly_s1 = (2 * weekly_pivot) - high_1d
    # Resistance 1: 2*Pivot - Low
    weekly_r1 = (2 * weekly_pivot) - low_1d
    
    # Align to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    
    # Determine pivot bias: price above/below weekly pivot
    pivot_bias = np.where(close_1d > weekly_pivot, 1, -1)
    pivot_bias_aligned = align_htf_to_ltf(prices, df_1d, pivot_bias)
    
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
            np.isnan(pivot_bias_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require pivot bias alignment (weekly directional filter)
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Long: price breaks above Donchian high AND pivot bias bullish AND price above weekly pivot
            if (price > donch_high[i] and 
                pivot_bias_aligned[i] > 0 and 
                price > weekly_pivot_aligned[i]):
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below Donchian low AND pivot bias bearish AND price below weekly pivot
            elif (price < donch_low[i] and 
                  pivot_bias_aligned[i] < 0 and 
                  price < weekly_pivot_aligned[i]):
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