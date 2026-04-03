#!/usr/bin/env python3
"""
Experiment #271: 6h Alligator + Elder Ray from 1d trend (MTF)
HYPOTHESIS: 6h Williams Alligator (JAW/TEETH/LIPS) identifies trend state, while 1d Elder Ray (Bull/Bear Power) filters for alignment with higher-timeframe trend. Enter long when Alligator is bullish (Lips>Teeth>Jaw) AND 1d Bull Power > 0; short when Alligator bearish (Lips<Teeth<Jaw) AND 1d Bear Power < 0. Uses ATR(6h) for dynamic stoploss. Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing (0.25) to minimize fee drag. Works in bull via trend continuation and bear via filtering out counter-trend swings.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_271_6h_alligator_elder_ray_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Elder Ray (Bull/Bear Power) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 13-period EMA on 1d close
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    # Align to 6h timeframe (shifted by align_htf_to_ltf for completed bars)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # === 6h Indicators: Williams Alligator (SMMA) ===
    # Alligator uses Smoothed Moving Average (SMMA) with periods 13,8,5 and offsets 8,5,3
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=np.float64)
        result = np.full_like(arr, np.nan, dtype=np.float64)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high, 13)  # Jaw (Blue) - 13-period SMMA of Median Price, offset 8 bars
    teeth = smma(high, 8)  # Teeth (Red) - 8-period SMMA, offset 5 bars
    lips = smma(high, 5)   # Lips (Green) - 5-period SMMA, offset 3 bars
    
    # Apply offsets (shift right by offset bars)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # First offset bars become NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Enough for Alligator offsets and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(atr_14[i]) or np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Alligator Trend State ---
        # Bullish: Lips > Teeth > Jaw (Green > Red > Blue)
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish: Lips < Teeth < Jaw (Green < Red < Blue)
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # --- Elder Ray Trend Filter from 1d ---
        # Bull Power > 0 indicates bullish momentum
        # Bear Power < 0 indicates bearish momentum
        elder_bullish = bull_power_aligned[i] > 0
        elder_bearish = bear_power_aligned[i] < 0
        
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
                # Exit if Alligator turns bearish OR Elder Ray turns bearish
                if alligator_bearish or not elder_bullish:
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
                # Exit if Alligator turns bullish OR Elder Ray turns bullish
                if alligator_bullish or not elder_bearish:
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
        # Long: Alligator bullish AND Elder Ray bullish
        if alligator_bullish and elder_bullish:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Alligator bearish AND Elder Ray bearish
        elif alligator_bearish and elder_bearish:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals