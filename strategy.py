#!/usr/bin/env python3
"""
Experiment #175: 6h Williams Alligator + Elder Ray + 1d Trend Filter
HYPOTHESIS: Williams Alligator (jaw/teeth/lips) identifies trend absence/presence, Elder Ray measures bull/bear power strength, and 1d EMA200 filter ensures alignment with higher timeframe trend. This combination works in both bull/bear markets by only taking trades when all three indicators agree, reducing false signals. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_175_6h_alligator_elder_ray_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA200 trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 6h Indicators: Williams Alligator (13,8,5) smoothed with SMA ===
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            result[period-1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high + low, 13)  # Using median price
    jaw = np.roll(jaw, 8)       # Shift 8 bars forward
    teeth = smma(high + low, 8) 
    teeth = np.roll(teeth, 5)   # Shift 5 bars forward
    lips = smma(high + low, 5)
    lips = np.roll(lips, 3)     # Shift 3 bars forward
    
    # === 6h Indicators: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) ===
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
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
    
    warmup = 200  # For EMA200
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr_14[i]) or np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Alligator Trend Detection ---
        # Alligator asleep (no trend): jaws, teeth, lips intertwined
        alligator_asleep = (abs(jaw[i] - teeth[i]) < (price * 0.001) and 
                           abs(teeth[i] - lips[i]) < (price * 0.001) and
                           abs(lips[i] - jaw[i]) < (price * 0.001))
        # Alligator awake (trending): lips > teeth > jaw (up) OR lips < teeth < jaw (down)
        alligator_up = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_down = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # --- Elder Ray Strength ---
        strong_bull = bull_power[i] > 0 and bull_power[i] > bear_power[i] * (-1)
        strong_bear = bear_power[i] < 0 and abs(bear_power[i]) > bull_power[i]
        
        # --- 1d Trend Filter ---
        uptrend_1d = price > ema_200_1d_aligned[i]
        downtrend_1d = price < ema_200_1d_aligned[i]
        
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
                # Exit if alligator goes back to sleep or trend weakens
                if alligator_asleep or (not alligator_up and not strong_bull):
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
                # Exit if alligator goes back to sleep or trend weakens
                if alligator_asleep or (not alligator_down and not strong_bear):
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
        # Long: Alligator up + strong bull power + 1d uptrend
        if alligator_up and strong_bull and uptrend_1d:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Alligator down + strong bear power + 1d downtrend
        elif alligator_down and strong_bear and downtrend_1d:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals