#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + 1w Elder Ray Bull Power confirmation + volume spike.
Long when Alligator jaws (13-period SMMA) crosses above teeth (8-period SMMA) AND 1w Bull Power > 0 AND volume > 2.0x 20-period average.
Short when Alligator jaws crosses below teeth AND 1w Bear Power < 0 AND volume > 2.0x 20-period average.
Exit when Alligator jaws crosses back below teeth (for long) or above teeth (for short) OR ATR stoploss (2.0*ATR).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year (50-150 total over 4 years).
Alligator identifies trend, Elder Ray confirms 1w momentum, volume filters noise, ATR stop controls risk.
Works in bull/bear by requiring 1w Elder Ray alignment (avoids counter-trend trades in strong regimes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, period):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if len(source) < period:
        return np.full_like(source, np.nan, dtype=np.float64)
    result = np.full_like(source, np.nan, dtype=np.float64)
    # First value is simple average
    result[period-1] = np.mean(source[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
    for i in range(period, len(source)):
        result[i] = (result[i-1] * (period-1) + source[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Calculate 12h Williams Alligator (Jaws=13, Teeth=8, Lips=5 SMMA)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    
    jaws = smma(median_price_12h, 13)  # Jaws (Blue) - 13-period SMMA
    teeth = smma(median_price_12h, 8)   # Teeth (Red) - 8-period SMMA
    lips = smma(median_price_12h, 5)    # Lips (Green) - 5-period SMMA
    
    # Align Alligator lines to 12h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate 1w Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    ema_1w_13 = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1w - ema_1w_13  # Bull Power = High - EMA13
    bear_power = low_1w - ema_1w_13   # Bear Power = Low - EMA13
    
    # Align Elder Ray to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1w, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1w, bear_power)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 13, 13, 5, 20, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        jaws_val = jaws_aligned[i]
        teeth_val = teeth_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        
        if position == 0:
            # Long: Jaws crosses above Teeth AND Bull Power > 0 AND volume spike
            if (jaws_val > teeth_val and jaws_aligned[i-1] <= teeth_aligned[i-1] and  # crossover
                bull_power_val > 0 and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Jaws crosses below Teeth AND Bear Power < 0 AND volume spike
            elif (jaws_val < teeth_val and jaws_aligned[i-1] >= teeth_aligned[i-1] and  # crossover
                  bear_power_val < 0 and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Jaws crosses back below Teeth (for long) or above Teeth (for short)
            if position == 1 and jaws_val < teeth_val and jaws_aligned[i-1] >= teeth_aligned[i-1]:
                exit_signal = True
            elif position == -1 and jaws_val > teeth_val and jaws_aligned[i-1] <= teeth_aligned[i-1]:
                exit_signal = True
            
            # ATR-based stoploss: 2.0 * ATR from entry
            if position == 1 and price < entry_price - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1wElderRay_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0