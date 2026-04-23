#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Williams Alligator (Jaw/Teeth/Lips) with 1w Elder Ray Bull/Bear Power and volume confirmation.
Long when price > Alligator Lips AND Bull Power > 0 AND volume > 1.5x 20-period average.
Short when price < Alligator Jaw AND Bear Power < 0 AND volume > 1.5x 20-period average.
Exit when price crosses Alligator Teeth or ATR stoploss hit (2.5*ATR).
Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.
Designed for 4h timeframe to target 20-50 trades/year per symbol (80-200 total over 4 years).
Williams Alligator identifies trend absence/presence; Elder Ray measures bull/bear power behind moves.
Works in both bull and bear markets by trading with the 1w Elder Ray trend and using volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_alligator, compute_elder_ray

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
    
    # Calculate 1d Williams Alligator (Jaw=13, Teeth=8, Lips=5)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    median_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    jaw, teeth, lips = compute_williams_alligator(median_1d, 13, 8, 5)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw, additional_delay_bars=0)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth, additional_delay_bars=0)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips, additional_delay_bars=0)
    
    # Calculate 1w Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_13 = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    bull_power = high_1w - ema_1w_13
    bear_power = low_1w - ema_1w_13
    bull_power_aligned = align_htf_to_ltf(prices, df_1w, bull_power, additional_delay_bars=0)
    bear_power_aligned = align_htf_to_ltf(prices, df_1w, bear_power, additional_delay_bars=0)
    
    # Volume average (20-period) on 4h timeframe
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
    start_idx = max(100, 13, 13, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
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
        jaw_val = jaw_aligned[i]
        lips_val = lips_aligned[i]
        teeth_val = teeth_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        
        if position == 0:
            # Long: Price > Alligator Lips AND Bull Power > 0 AND volume spike
            if (price > lips_val and 
                bull_power_val > 0 and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price < Alligator Jaw AND Bear Power < 0 AND volume spike
            elif (price < jaw_val and 
                  bear_power_val < 0 and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price crosses Alligator Teeth
            if position == 1 and price < teeth_val:
                exit_signal = True
            elif position == -1 and price > teeth_val:
                exit_signal = True
            
            # ATR-based stoploss: 2.5 * ATR from entry
            if position == 1 and price < entry_price - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsAlligator_1wElderRay_VolumeConfirmation_ATRStop"
timeframe = "4h"
leverage = 1.0