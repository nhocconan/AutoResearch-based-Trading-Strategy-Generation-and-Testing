#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1d Williams Alligator (Jaw/Teeth/Lips) with volume confirmation and chop regime filter.
Long when price > Alligator Lips AND Alligator is bullish (Lips > Teeth > Jaw) AND volume > 1.5x 20-period average AND chop < 61.8 (trending regime).
Short when price < Alligator Lips AND Alligator is bearish (Lips < Teeth < Jaw) AND volume > 1.5x 20-period average AND chop < 61.8 (trending regime).
Exit when price crosses Alligator Teeth or ATR stoploss hit (2.5*ATR).
Uses discrete position sizing (0.25) to minimize fee churn.
Designed for 12h timeframe to target 12-37 trades/year per symbol (50-150 total over 4 years).
Works in both bull and bear markets by using chop regime filter to avoid ranging markets and Alligator for trend confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Alligator: Jaw (13-period SMMA, shifted 8), Teeth (8-period SMMA, shifted 5), Lips (5-period SMMA, shifted 3)
    # SMMA = Smoothed Moving Average (similar to Wilder's EMA with alpha=1/period)
    def smma(arr, period):
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        result = np.full(len(arr), np.nan)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Shift: Jaw by 8, Teeth by 5, Lips by 3
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set shifted values to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Choppiness Index (14-period) on 1d timeframe
    def choppiness_index(high, low, close, period=14):
        if len(high) < period:
            return np.full(len(high), np.nan)
        atr = np.zeros(len(high))
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        for i in range(1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period  # Wilder's smoothing
        # Sum of ATR over period
        atr_sum = np.zeros(len(high))
        for i in range(period-1, len(high)):
            atr_sum[i] = np.sum(atr[i-period+1:i+1])
        # Highest high and lowest low over period
        hh = np.zeros(len(high))
        ll = np.zeros(len(high))
        for i in range(period-1, len(high)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        # Choppiness Index formula
        chop = np.zeros(len(high))
        for i in range(period-1, len(high)):
            if hh[i] - ll[i] != 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
            else:
                chop[i] = 0
        # Set values before period to NaN
        chop[:period-1] = np.nan
        return chop
    
    chop = choppiness_index(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation on 12h timeframe
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
    start_idx = max(20, 14, 13+8, 8+5, 5+3, 14)  # 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        chop_val = chop_aligned[i]
        
        # Regime filter: only trade in trending markets (chop < 61.8)
        is_trending = chop_val < 61.8
        
        if position == 0:
            # Long: Price > Lips AND Alligator bullish (Lips > Teeth > Jaw) AND volume spike AND trending regime
            if (price > lips_val and lips_val > teeth_val and teeth_val > jaw_val and 
                volume[i] > 1.5 * vol_ma_val and is_trending):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price < Lips AND Alligator bearish (Lips < Teeth < Jaw) AND volume spike AND trending regime
            elif (price < lips_val and lips_val < teeth_val and teeth_val < jaw_val and 
                  volume[i] > 1.5 * vol_ma_val and is_trending):
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

name = "12H_WilliamsAlligator_VolumeConfirmation_ChopFilter_ATRStop"
timeframe = "12h"
leverage = 1.0