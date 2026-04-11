#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + Volume confirmation
# - Uses Williams Alligator (JAW=13, TEETH=8, LIPS=5) smoothed with SMMA to identify trend direction
# - Elder Ray Index (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures trend strength
# - Volume confirmation: current volume > 1.5x 20-period average volume
# - Long: Alligator aligned (Lips > Teeth > Jaw) + Bull Power > 0 + volume confirmation
# - Short: Alligator aligned inversely (Lips < Teeth < Jaw) + Bear Power < 0 + volume confirmation
# - Exit: Opposite Alligator alignment or Elder Ray power crosses zero
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Works in both bull and bear markets by capturing strong trending moves with volume confirmation

name = "12h_1d_alligator_elder_ray_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Elder Ray and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 1d Elder Ray components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema_13_1d  # Bull Power = High - EMA13
    bear_power_1d = low_1d - ema_13_1d   # Bear Power = Low - EMA13
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Williams Alligator on 12h timeframe (smoothed with SMMA)
    # JAW: 13-period SMMA, shifted 8 bars
    # TEETH: 8-period SMMA, shifted 5 bars
    # LIPS: 5-period SMMA, shifted 3 bars
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CLOSE) / N
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that don't have enough data
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(bull_power_1d[i]) or np.isnan(bear_power_1d[i]) or np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        volume_current = volume[i]
        
        # Alligator alignment
        lips_val = lips_shifted[i]
        teeth_val = teeth_shifted[i]
        jaw_val = jaw_shifted[i]
        
        # Elder Ray components from 1d (aligned)
        bull_power = bull_power_1d[i]
        bear_power = bear_power_1d[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Alligator aligned upward (Lips > Teeth > Jaw) + Bull Power > 0 + volume confirmation
        if lips_val > teeth_val > jaw_val and bull_power > 0 and vol_confirm:
            enter_long = True
        
        # Short: Alligator aligned downward (Lips < Teeth < Jaw) + Bear Power < 0 + volume confirmation
        if lips_val < teeth_val < jaw_val and bear_power < 0 and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Alligator alignment breaks or Bull Power turns negative
            exit_long = not (lips_val > teeth_val > jaw_val) or bull_power <= 0
        elif position == -1:
            # Exit short if Alligator alignment breaks or Bear Power turns positive
            exit_short = not (lips_val < teeth_val < jaw_val) or bear_power >= 0
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals