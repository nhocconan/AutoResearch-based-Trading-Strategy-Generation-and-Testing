#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) with 1w EMA50 trend filter and volume confirmation
# Uses 6h for Alligator (JAW=13, TEETH=8, LIPS=5 SMMA) and Elder Ray (EMA13) calculations
# 1w EMA50 for major trend filter (only trade in direction of weekly trend)
# Volume confirmation (1.5x 50-period average on 6h) ensures institutional participation
# Session filter (08-20 UTC) reduces noise trades outside active hours
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Alligator identifies trend absence (all lines intertwined) vs presence (lines diverged)
# Elder Ray measures bull/bear power relative to EMA13
# Designed for low trade frequency to minimize fee drag (critical for 6h timeframe)

name = "6h_Alligator_ElderRay_1wEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator on 6h data (SMMA = smoothed moving average)
    # Jaw (blue): 13-period SMMA, shifted 8 bars ahead
    # Teeth (red): 8-period SMMA, shifted 5 bars ahead  
    # Lips (green): 5-period SMMA, shifted 3 bars ahead
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # 13-period SMMA
    teeth = smma(close, 8)  # 8-period SMMA
    lips = smma(close, 5)   # 5-period SMMA
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)   # shifted 8 bars ahead
    teeth = np.roll(teeth, 5) # shifted 5 bars ahead
    lips = np.roll(lips, 3)   # shifted 3 bars ahead
    
    # Calculate Elder Ray (Bull/Bear Power) on 6h data
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation (1.5x 50-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator sleeping condition: all lines intertwined (market ranging)
            # Alligator awake condition: lines separated and ordered (trending)
            # For longs: Lips > Teeth > Jaw (green above red above blue) AND Bull Power > 0
            # For shorts: Lips < Teeth < Jaw (green below red below blue) AND Bear Power < 0
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and  # Alligator awake - bullish alignment
                bull_power[i] > 0 and                          # Bullish power
                close[i] > ema_50_1w_aligned[i] and            # Above weekly EMA50 (uptrend filter)
                volume_confirm[i]):                            # Volume confirmation
                signals[i] = 0.25
                position = 1
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and  # Alligator awake - bearish alignment
                  bear_power[i] < 0 and                         # Bearish power
                  close[i] < ema_50_1w_aligned[i] and           # Below weekly EMA50 (downtrend filter)
                  volume_confirm[i]):                           # Volume confirmation
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator starts to sleep (lines intertwine) OR Bear Power becomes positive
            # Lines intertwine when Teeth is between Lips and Jaw (not strictly ordered)
            if not ((lips[i] > teeth[i] and teeth[i] > jaw[i]) or  # Bullish alignment
                    (lips[i] < teeth[i] and teeth[i] < jaw[i])):   # Bearish alignment
                signals[i] = 0.0
                position = 0
            elif bear_power[i] > 0:  # Bear power turned positive (bulls losing control)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator starts to sleep (lines intertwine) OR Bull Power becomes negative
            if not ((lips[i] > teeth[i] and teeth[i] > jaw[i]) or  # Bullish alignment
                    (lips[i] < teeth[i] and teeth[i] < jaw[i])):   # Bearish alignment
                signals[i] = 0.0
                position = 0
            elif bull_power[i] < 0:  # Bull power turned negative (bears losing control)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals