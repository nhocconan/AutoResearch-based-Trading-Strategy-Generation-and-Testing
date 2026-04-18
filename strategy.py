#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Williams Alligator + Volume Spike
# Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength.
# ADX > 25 confirms trending market, avoiding whipsaws in ranging conditions.
# Volume spike confirms institutional participation in the breakout.
# Works in bull markets (long when price > Teeth, ADX>25, volume spike) and bear markets (short when price < Teeth, ADX>25, volume spike).
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
name = "6h_ADX_WilliamsAlligator_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator from 1d data (using smoothed SMAs)
    # Alligator Jaw: 13-period SMMA, shifted 8 bars forward
    # Alligator Teeth: 8-period SMMA, shifted 5 bars forward
    # Alligator Lips: 5-period SMMA, shifted 3 bars forward
    close_1d = df_1d['close'].values
    
    # Calculate SMMA (Smoothed Moving Average) - equivalent to Wilder's smoothing
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(close_1d, 13)
    teeth_raw = smma(close_1d, 8)
    lips_raw = smma(close_1d, 5)
    
    # Shift forward: Jaw by 8, Teeth by 5, Lips by 3
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw_raw) > 8:
        jaw[8:] = jaw_raw[:-8]
    if len(teeth_raw) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips_raw) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Calculate ADX from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def WilderSmoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple sum
        result[period-1] = np.sum(arr[:period])
        # Subsequent values: prev - (prev/period) + current
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr_period = WilderSmoothing(tr, 14)
    plus_dm_period = WilderSmoothing(plus_dm, 14)
    minus_dm_period = WilderSmoothing(minus_dm, 14)
    
    # DI values
    plus_di = 100 * plus_dm_period / tr_period
    minus_di = 100 * minus_dm_period / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = WilderSmoothing(dx, 14)
    
    # Align Alligator lines and ADX to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate volume spike: current volume > 2.0 * 12-period average volume (3 days on 6h chart)
    vol_ma_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_spike = volume > (2.0 * vol_ma_12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_12[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Long: Price above Teeth, Alligator aligned (Lips > Teeth > Jaw), ADX>25, volume spike
            if (close_val > teeth_val and lips_val > teeth_val and teeth_val > jaw_val and
                adx_val > 25 and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below Teeth, Alligator aligned (Lips < Teeth < Jaw), ADX>25, volume spike
            elif (close_val < teeth_val and lips_val < teeth_val and teeth_val < jaw_val and
                  adx_val > 25 and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price below Lips (trend weakness) or ADX < 20 (trend ending)
            if close_val < lips_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price above Lips (trend weakness) or ADX < 20 (trend ending)
            if close_val > lips_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals