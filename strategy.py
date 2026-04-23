#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d/1w regime filter and volume confirmation.
Long when Jaw < Teeth < Lips (bullish alignment) AND price > Lips AND volume > 1.5x 20-period average.
Short when Jaw > Teeth > Lips (bearish alignment) AND price < Lips AND volume > 1.5x 20-period average.
Exit when Alligator lines cross (alignment breaks) OR ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) targeting ~20 trades/year on 12h timeframe.
Williams Alligator identifies trending vs ranging markets; we only trade in strong trends with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1w ADX25 for regime filter (trending market)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                       np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        if len(values) < period:
            return result
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    tr14 = wilders_smoothing(tr_1w, 14)
    dm_plus14 = wilders_smoothing(dm_plus, 14)
    dm_minus14 = wilders_smoothing(dm_minus, 14)
    
    # DI and DX
    di_plus = np.where(tr14 != 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus14 / tr14, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Williams Alligator on 12h timeframe (using SMMA)
    def smoothed_moving_average(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        sma = np.mean(values[:period])
        result = np.full(len(values), np.nan)
        result[period-1] = sma
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smoothed_moving_average(close, 13)  # Jaw (Blue) - 13-period SMMA
    teeth = smoothed_moving_average(close, 8)   # Teeth (Red) - 8-period SMMA
    lips = smoothed_moving_average(close, 5)    # Lips (Green) - 5-period SMMA
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13, 8, 5, 20, 14)  # EMA50 needs 50, Alligator needs 13, vol MA needs 20, ATR needs 14, ADX needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema50_val = ema50_1d_aligned[i]
        adx_val = adx_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        if position == 0:
            # Long: Bullish Alligator alignment (Jaw < Teeth < Lips) AND price > Lips AND volume > 1.5x avg AND trending regime (ADX > 25) AND price > EMA50
            if (jaw_val < teeth_val < lips_val and 
                price > lips_val and 
                volume[i] > 1.5 * vol_ma_val and 
                adx_val > 25 and 
                price > ema50_val):
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Bearish Alligator alignment (Jaw > Teeth > Lips) AND price < Lips AND volume > 1.5x avg AND trending regime (ADX > 25) AND price < EMA50
            elif (jaw_val > teeth_val > lips_val and 
                  price < lips_val and 
                  volume[i] > 1.5 * vol_ma_val and 
                  adx_val > 25 and 
                  price < ema50_val):
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Alligator alignment breaks (lines cross)
            if position == 1 and not (jaw_val < teeth_val < lips_val):
                exit_signal = True
            elif position == -1 and not (jaw_val > teeth_val > lips_val):
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Williams_Alligator_1dEMA50_1wADX25_VolumeConfirmation_AlignmentExit_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0