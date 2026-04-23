#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
Long when Alligator jaws (13-period SMMA) > teeth (8-period SMMA) > lips (5-period SMMA) AND 1d EMA34 is rising AND volume > 1.5x 20-period average.
Short when jaws < teeth < lips AND 1d EMA34 is falling AND volume > 1.5x 20-period average.
Exit when Alligator lines crossover in opposite direction or ATR stoploss hit (2.5*ATR).
Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.
Targets 12-37 trades/year per symbol (50-150 total over 4 years) by requiring strong Alligator alignment and volume confirmation.
Designed to work in both bull and bear markets by trading with the Alligator's trending phase and using the 1d EMA for higher timeframe trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA)"""
    if length < 1:
        return source
    smma = np.full_like(source, np.nan, dtype=np.float64)
    smma[length-1] = np.mean(source[:length])
    for i in range(length, len(source)):
        smma[i] = (smma[i-1] * (length-1) + source[i]) / length
    return smma

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
    
    # Calculate Williams Alligator from 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    jaws = smma(median_12h, 13)
    teeth = smma(median_12h, 8)
    lips = smma(median_12h, 5)
    
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_1d_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # EMA slope (rising/falling) - compare current vs 1 period ago
    ema_slope = np.zeros_like(ema_1d_34_aligned)
    ema_slope[1:] = ema_1d_34_aligned[1:] - ema_1d_34_aligned[:-1]
    
    # Volume average (20-period) on 12h timeframe
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
    start_idx = max(100, 34, 20, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_1d_34_aligned[i]) or np.isnan(ema_slope[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
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
        lips_val = lips_aligned[i]
        ema_slope_val = ema_slope[i]
        
        if position == 0:
            # Long: Jaws > Teeth > Lips (Alligator bullish alignment) AND 1d EMA34 rising AND volume spike
            if (jaws_val > teeth_val > lips_val and 
                ema_slope_val > 0 and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Jaws < Teeth < Lips (Alligator bearish alignment) AND 1d EMA34 falling AND volume spike
            elif (jaws_val < teeth_val < lips_val and 
                  ema_slope_val < 0 and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Alligator lines crossover in opposite direction
            if position == 1 and jaws_val < teeth_val:
                exit_signal = True
            elif position == -1 and jaws_val > teeth_val:
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

name = "12H_WilliamsAlligator_1dEMA34_Trend_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0