#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray Power + 1w EMA50 trend filter
# Uses Williams Alligator (jaw/teeth/lips) from 6h for trend state and entry timing
# Elder Ray (Bull/Bear power) from 1d confirms institutional buying/selling pressure
# 1w EMA50 filter ensures we only trade in alignment with weekly supertrend
# Volume confirmation (1.5x 24-period average) filters low-participation breakouts
# Designed for low trade frequency (12-30/year) to minimize fee drag on 6h timeframe
# Works in bull markets via trend-following entries, in bear via power divergence signals

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
    
    # Load 6h data ONCE before loop for Alligator indicator
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 6h
    # Jaw (Blue): 13-period SMMA, shifted 8 bars ahead
    # Teeth (Red): 8-period SMMA, shifted 5 bars ahead  
    # Lips (Green): 5-period SMMA, shifted 3 bars ahead
    close_6h = df_6h['close'].values
    # SMMA calculation (smoothed moving average)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_6h, 13)
    teeth = smma(close_6h, 8)
    lips = smma(close_6h, 5)
    
    # Shift as per Alligator specification
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to LTF (6h)
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # Load 1d data ONCE before loop for Elder Ray Power
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Elder Ray Power (Bull/Bear) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 13-period EMA of close
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_1d - ema_13_1d
    # Bear Power = Low - EMA13
    bear_power = low_1d - ema_13_1d
    
    # Align Elder Ray to LTF (6h)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation (1.5x 24-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
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
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        alligator_bull = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        alligator_bear = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray confirmation: Bull Power > 0 and rising, Bear Power < 0 and falling
        bull_confirm = bull_power_aligned[i] > 0 and (i == start_idx or bull_power_aligned[i] > bull_power_aligned[i-1])
        bear_confirm = bear_power_aligned[i] < 0 and (i == start_idx or bear_power_aligned[i] < bear_power_aligned[i-1])
        
        # Weekly trend filter: price above/below 1w EMA50
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator bullish + Bull Power confirming + weekly uptrend + volume
            if alligator_bull and bull_confirm and weekly_uptrend and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish + Bear Power confirming + weekly downtrend + volume
            elif alligator_bear and bear_confirm and weekly_downtrend and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish OR Bull Power turns negative
            if not alligator_bull or bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish OR Bear Power turns positive
            if not alligator_bear or bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals