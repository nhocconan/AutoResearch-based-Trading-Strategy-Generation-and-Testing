#!/usr/bin/env python3
"""
Experiment #131: 6h Williams Alligator + 1d Elder Ray Trend Filter
HYPOTHESIS: Williams Alligator (jaw/teeth/lips) identifies trend absence/presence on 6h, while 1d Elder Ray (Bull/Bear Power) confirms higher-timeframe trend direction. Only trade when Alligator is 'awake' (trending) and Elder Ray aligns with the direction. This avoids whipsaws in ranging markets and captures strong trends in both bull and bear markets. Discrete sizing (0.25) and ATR stoploss (2.5x) control risk and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_131_6h_alligator_1d_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Elder Ray (Bull/Bear Power) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 13-period EMA for Elder Ray
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    # Align to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # === 6h Indicators: Williams Alligator ===
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(src, length):
        result = np.full_like(src, np.nan, dtype=np.float64)
        if len(src) < length:
            return result
        # First value is SMA
        result[length-1] = np.mean(src[:length])
        # Subsequent values: SMMA = (Prev SMMA * (length-1) + Current Price) / length
        for i in range(length, len(src)):
            result[i] = (result[i-1] * (length-1) + src[i]) / length
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8) # Red line
    lips = smma(close, 5)  # Green line
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # First values become NaN after roll
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # === 6h Indicators: ATR(20) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 60
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(atr[i]) or np.isnan(bull_power_1d_aligned[i]) or
            np.isnan(bear_power_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Alligator Trend Detection ---
        # Alligator is 'asleep' (no trend) when lines are intertwined
        # Alligator is 'awake' (trending) when lips > teeth > jaw (up) or lips < teeth < jaw (down)
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        trending_up = lips_above_teeth and teeth_above_jaw
        trending_down = lips_below_teeth and teeth_below_jaw
        
        # --- Elder Ray Trend Filter (1d) ---
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        bullish_htf = bull_power > 0  # Higher timeframe bullish
        bearish_htf = bear_power < 0  # Higher timeframe bearish
        
        # --- Entry Logic ---
        long_entry = trending_up and bullish_htf
        short_entry = trending_down and bearish_htf
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level or not (lips_above_teeth and teeth_above_jaw):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level or not (lips_below_teeth and teeth_below_jaw):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry ---
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals