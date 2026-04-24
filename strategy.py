#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA50 trend filter and volume confirmation.
- Williams Alligator uses smoothed medians (13,8,5) to identify trends: Lips > Teeth > Jaw = uptrend, reverse = downtrend.
- 1d EMA50 trend filter ensures alignment with daily momentum, reducing counter-trend trades.
- Volume spike (>1.8x 20-period average) confirms breakout validity with moderate threshold.
- Discrete position sizing (0.25) balances return and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
- Uses 1d HTF data loaded ONCE before loop per MTF rules.
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
    
    # Get 1d data ONCE before loop for Williams Alligator and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator from 1d data
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (df_1d['high'] + df_1d['low']) / 2
    close_1d = df_1d['close'].values
    
    # Smoothed Moving Average (SMMA) calculation
    def smma(series, period):
        if len(series) < period:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        # First value is SMA
        result[period-1] = np.mean(series[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT_VALUE) / PERIOD
        for i in range(period, len(series)):
            result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    jaw_raw = smma(median_price.values, 13)
    teeth_raw = smma(median_price.values, 8)
    lips_raw = smma(median_price.values, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Align Williams Alligator lines to 12h timeframe (using previous completed 1d bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 1.8x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) with volume spike and above 1d EMA50
            if lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and volume_spike[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) with volume spike and below 1d EMA50
            elif lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and volume_spike[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw) OR below 1d EMA50
            if lips_aligned[i] <= teeth_aligned[i] or teeth_aligned[i] <= jaw_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw) OR above 1d EMA50
            if lips_aligned[i] >= teeth_aligned[i] or teeth_aligned[i] >= jaw_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0