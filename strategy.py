#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using Williams Alligator (Jaw/Teeth/Lips) for trend direction
combined with 1d ATR-based volatility filter and volume confirmation. Long when Alligator
is bullish (Lips > Teeth > Jaw) with price above Teeth, volume > 1.5x average, and 1d ATR
below its 20-period median (low volatility). Short when Alligator is bearish (Lips < Teeth < Jaw)
with price below Teeth, volume confirmation, and low 1d volatility. Uses discrete position sizing
to minimize fee churn and targets 50-150 trades over 4 years. Works in both bull and bear markets
by requiring alignment with higher timeframe trend and volatility regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (used in Williams Alligator)"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    sma = pd.Series(arr).rolling(window=period, min_periods=period).mean().values
    smma = np.full_like(arr, np.nan)
    smma[period-1] = sma[period-1]
    for i in range(period, len(arr)):
        smma[i] = (smma[i-1] * (period-1) + arr[i]) / period
    return smma

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ATR and Alligator calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 1d timeframe
    # Jaw: 13-period SMMA smoothed by 8
    jaw_raw = smma(close_1d, 13)
    jaw = smma(jaw_raw, 8) if not np.all(np.isnan(jaw_raw)) else np.full_like(close_1d, np.nan)
    # Teeth: 8-period SMMA smoothed by 5
    teeth_raw = smma(close_1d, 8)
    teeth = smma(teeth_raw, 5) if not np.all(np.isnan(teeth_raw)) else np.full_like(close_1d, np.nan)
    # Lips: 5-period SMMA smoothed by 3
    lips_raw = smma(close_1d, 5)
    lips = smma(lips_raw, 3) if not np.all(np.isnan(lips_raw)) else np.full_like(close_1d, np.nan)
    
    # Calculate 1d ATR(14) and its 20-period median for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.concatenate([[np.nan], tr])
    atr_ma_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
    atr_median_1d = pd.Series(atr_ma_1d).rolling(window=20, min_periods=20).median().values
    
    # Align HTF indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    atr_median_aligned = align_htf_to_ltf(prices, df_1d, atr_median_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(atr_median_aligned[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        atr_median_val = atr_median_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        # Check low volatility condition (1d ATR below its median)
        low_volatility = atr_median_val > 0 and atr_ma_1d[i] < atr_median_val if not np.isnan(atr_ma_1d[i]) else False
        
        if position == 0:
            # Bullish Alligator: Lips > Teeth > Jaw
            bullish = lips_val > teeth_val and teeth_val > jaw_val
            # Bearish Alligator: Lips < Teeth < Jaw
            bearish = lips_val < teeth_val and teeth_val < jaw_val
            
            # Long: bullish Alligator AND price > Teeth AND volume confirmation AND low volatility
            if bullish and price > teeth_val and vol_current > 1.5 * vol_ma_val and low_volatility:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator AND price < Teeth AND volume confirmation AND low volatility
            elif bearish and price < teeth_val and vol_current > 1.5 * vol_ma_val and low_volatility:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator reversal OR high volatility
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator turns bearish OR price crosses below Jaw OR high volatility
                bearish = lips_val < teeth_val and teeth_val < jaw_val
                if bearish or price < jaw_val or not low_volatility:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Alligator turns bullish OR price crosses above Jaw OR high volatility
                bullish = lips_val > teeth_val and teeth_val > jaw_val
                if bullish or price > jaw_val or not low_volatility:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dATR_Volume"
timeframe = "12h"
leverage = 1.0