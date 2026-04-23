#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above Alligator Jaw (13-period SMMA) AND 1d EMA50 is rising AND volume > 2.0x 20-period average.
Short when price breaks below Alligator Jaw AND 1d EMA50 is falling AND volume > 2.0x 20-period average.
Exit when price crosses the Alligator Teeth (8-period SMMA) or EMA50 direction reverses.
Uses 1d HTF for EMA50 trend to reduce whipsaws in bear markets. Target: 50-150 total trades over 4 years (12-37/year).
Williams Alligator: Jaw=13 SMMA, Teeth=8 SMMA, Lips=5 SMMA (all smoothed with 8,5,3 periods respectively).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if len(values) < period:
        return np.full(len(values), np.nan)
    result = np.full(len(values), np.nan)
    # First value is SMA
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Value) / Period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Williams Alligator (Jaw, Teeth, Lips) for entry signals
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Jaw: 13-period SMMA
    jaw_12h = smma(close_12h, 13)
    # Teeth: 8-period SMMA
    teeth_12h = smma(close_12h, 8)
    # Lips: 5-period SMMA
    lips_12h = smma(close_12h, 5)
    
    # Align Alligator components to 12h timeframe (same as primary)
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 50, 20)  # Alligator Jaw (13), EMA50 (50), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        jaw = jaw_12h_aligned[i]
        teeth = teeth_12h_aligned[i]
        lips = lips_12h_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above Alligator Jaw AND EMA50 rising AND volume spike
            if price > jaw and ema_rising and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Alligator Jaw AND EMA50 falling AND volume spike
            elif price < jaw and ema_falling and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price crosses below Teeth OR EMA50 starts falling
                if price < teeth or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price crosses above Teeth OR EMA50 starts rising
                if price > teeth or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_JawBreakout_1dEMA50_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0