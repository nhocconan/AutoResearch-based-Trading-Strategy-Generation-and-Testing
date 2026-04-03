#!/usr/bin/env python3
"""
Experiment #079: 6h Alligator + Elder Ray (1d) Trend Strategy
HYPOTHESIS: Combines Williams Alligator (6h) for trend identification with Elder Ray (1d) for bull/bear power confirmation.
Alligator (jaw/teeth/lips) defines trend direction and strength; Elder Ray (Bull Power/Bear Power) filters entries to only
take trades where higher timeframe momentum supports the direction. Works in both bull/bear markets by only trading
in the direction of the 6h Alligator trend with 1d Elder Ray confirmation. Target: 75-150 trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_079_6h_alligator_elder_ray_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Indicators: Williams Alligator (6h) ===
    def calculate_smma(data, period):
        """Smoothed Moving Average (SMMA)"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=np.float64)
        result = np.full_like(data, np.nan, dtype=np.float64)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = calculate_smma(close, 13)  # Jaw: 13-period SMMA, shifted 8 bars
    teeth = calculate_smma(close, 8)  # Teeth: 8-period SMMA, shifted 5 bars
    lips = calculate_smma(close, 5)   # Lips: 5-period SMMA, shifted 3 bars
    
    # Apply Alligator shifts (jaw: +8, teeth: +5, lips: +3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set shifted values to NaN where invalid
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # === HTF: 1d Elder Ray for trend confirmation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values.astype(np.float64)
    low_1d = df_1d['low'].values.astype(np.float64)
    close_1d = df_1d['close'].values.astype(np.float64)
    
    # Calculate EMA(13) for Elder Ray
    def calculate_ema(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=np.float64)
        ema = np.full_like(data, np.nan, dtype=np.float64)
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    ema_13_1d = calculate_ema(close_1d, 13)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Warmup for Alligator and Elder Ray stability
    
    for i in range(warmup, n):
        # Skip if any critical values are NaN
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator trend detection:
        # Uptrend: Lips > Teeth > Jaw (all aligned upward)
        # Downtrend: Lips < Teeth < Jaw (all aligned downward)
        lips_val = lips_shifted[i]
        teeth_val = teeth_shifted[i]
        jaw_val = jaw_shifted[i]
        
        is_uptrend = lips_val > teeth_val and teeth_val > jaw_val
        is_downtrend = lips_val < teeth_val and teeth_val < jaw_val
        
        # Elder Ray confirmation:
        # Bull Power > 0 indicates bullish momentum
        # Bear Power < 0 indicates bearish momentum
        bull_confirm = bull_power_aligned[i] > 0
        bear_confirm = bear_power_aligned[i] < 0
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit when Alligator trend reverses or Elder Ray confirmation fails
            if position_side > 0:  # Long position
                if not (is_uptrend and bull_confirm):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if not (is_downtrend and bear_confirm):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 1 bar to prevent whipsaw
            if bars_since_entry < 1:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Long entry: Alligator uptrend AND Elder Ray bullish confirmation
        if is_uptrend and bull_confirm:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short entry: Alligator downtrend AND Elder Ray bearish confirmation
        elif is_downtrend and bear_confirm:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals