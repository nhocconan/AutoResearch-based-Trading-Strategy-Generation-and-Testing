#!/usr/bin/env python3
"""
1d Williams Alligator Breakout with Weekly EMA34 Trend and Volume Spike
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) on daily timeframe identifies trend structure. 
Breakout beyond Alligator's Lips with weekly EMA34 trend alignment and volume spike captures 
strong momentum moves. ATR trailing stop manages risk. Designed for ~20-50 trades/year on 1d 
to minimize fee drag and work in both bull/bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for volatility and trailing stop
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Williams Alligator on daily timeframe (MTF) - loaded ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    # Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
    close_1d = df_1d['close'].values
    # SMMA (Smoothed Moving Average) calculation
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) >= period:
            # First value is simple SMA
            result[period-1] = np.mean(arr[:period])
            # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)  # Jaw
    teeth = smma(close_1d, 8)  # Teeth
    lips = smma(close_1d, 5)   # Lips
    
    # Apply shifts: Jaw shifted 8 bars, Teeth shifted 5 bars, Lips shifted 3 bars
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Align to lower timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Weekly EMA34 trend filter (MTF)
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for all indicators
    start_idx = max(20, 34) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Alligator conditions: Lips above Teeth above Jaw = uptrend, Lips below Teeth below Jaw = downtrend
        alligator_uptrend = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        alligator_downtrend = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Breakout conditions: price breaks beyond Lips in direction of trend
        breakout_long = curr_close > lips_aligned[i]
        breakout_short = curr_close < lips_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Lips breakout + volume spike + weekly EMA34 trend alignment
            long_entry = breakout_long and vol_spike and alligator_uptrend and (curr_close > ema_34_1w_aligned[i])
            short_entry = breakout_short and vol_spike and alligator_downtrend and (curr_close < ema_34_1w_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management: ATR trailing stop
            highest_since_entry = max(highest_since_entry, curr_high)
            exit_level = highest_since_entry - (2.5 * atr_14[i])
            
            if curr_close < exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management: ATR trailing stop
            lowest_since_entry = min(lowest_since_entry, curr_low)
            exit_level = lowest_since_entry + (2.5 * atr_14[i])
            
            if curr_close > exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Williams_Alligator_Breakout_WeeklyEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0