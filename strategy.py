#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + 1d ADX Trend Filter + Volume Confirmation.
- Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) identifies trend direction and avoids choppy markets.
- 1d ADX > 25 ensures we only trade in strong trending conditions (both bull and bear markets).
- Volume spike (>1.8x 20-period average) confirms breakout validity and reduces false signals.
- Discrete position sizing (0.25) minimizes fee churn while allowing meaningful returns.
- Target trades: 50-150 total over 4 years (12-37/year) on 12h timeframe to avoid fee drag.
- Works in bull/bear markets via 1d ADX trend filter and Alligator's trend detection.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX calculation (trend strength filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period])  # skip index 0
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr_period = wilders_smoothing(tr, 14)
    plus_dm_period = wilders_smoothing(plus_dm, 14)
    minus_dm_period = wilders_smoothing(minus_dm, 14)
    
    # DI values
    plus_di = 100 * plus_dm_period / tr_period
    minus_di = 100 * minus_dm_period / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = np.full_like(dx, np.nan)
    adx[27:] = pd.Series(dx[27:]).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams Alligator on 12h data
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean()
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean()
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean()
    
    jaw_values = jaw.values
    teeth_values = teeth.values
    lips_values = lips.values
    
    # Volume confirmation: > 1.8x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(jaw_period, teeth_period, lips_period, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw_values[i]) or np.isnan(teeth_values[i]) or 
            np.isnan(lips_values[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator conditions: Jaw, Teeth, Lips alignment
        # Bullish: Lips > Teeth > Jaw (green alignment)
        # Bearish: Lips < Teeth < Jaw (red alignment)
        bullish_alligator = lips_values[i] > teeth_values[i] > jaw_values[i]
        bearish_alligator = lips_values[i] < teeth_values[i] < jaw_values[i]
        
        if position == 0:
            # Long: Bullish Alligator + ADX > 25 (strong trend) + volume spike
            if bullish_alligator and adx_aligned[i] > 25 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator + ADX > 25 (strong trend) + volume spike
            elif bearish_alligator and adx_aligned[i] > 25 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator turns bearish OR ADX falls below 20 (trend weakening)
            if not bullish_alligator or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator turns bullish OR ADX falls below 20 (trend weakening)
            if not bearish_alligator or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dADX_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0