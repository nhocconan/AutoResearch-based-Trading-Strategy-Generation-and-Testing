#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + 1d ADX trend filter + volume confirmation.
Long when price > Alligator Jaw AND Alligator Teeth > Alligator Lips (bullish alignment) AND ADX > 25 (trending) AND volume > 1.5x average.
Short when price < Alligator Jaw AND Alligator Teeth < Alligator Lips (bearish alignment) AND ADX > 25 AND volume > 1.5x average.
Exit when Alligator lines cross (Teeth crosses Jaw) OR ADX < 20 (range market).
Uses 6h for Alligator calculation and 1d for ADX/volume to reduce whipsaw.
Target: 50-150 total trades over 4 years (12-37/year). Alligator catches trends with smoothed lines,
ADX filters ranging markets, volume confirms breakout strength.
Works in bull markets (captures uptrends) and bear markets (captures downtrends).
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
    
    # Get 6h data for Alligator calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Alligator (Williams) on 6h timeframe
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(values, period):
        """Smoothed Moving Average"""
        sma = pd.Series(values).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full_like(values, np.nan, dtype=float)
        for i in range(len(values)):
            if i == period - 1:
                smma_vals[i] = sma[i]
            elif i >= period:
                smma_vals[i] = (smma_vals[i-1] * (period - 1) + values[i]) / period
        return smma_vals
    
    jaw_raw = smma(close_6h, 13)
    teeth_raw = smma(close_6h, 8)
    lips_raw = smma(close_6h, 5)
    
    # Apply shifts (Jaw: 8, Teeth: 5, Lips: 3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Get 1d data for ADX and volume filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d timeframe (14-period)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(values, period):
        """Wilder's smoothing (EMA with alpha=1/period)"""
        alpha = 1.0 / period
        ema = pd.Series(values).ewm(alpha=alpha, adjust=False, min_periods=period).mean().values
        # Set first 'period' values to NaN to match min_periods behavior
        ema[:period-1] = np.nan
        return ema
    
    atr_raw = wilders_smoothing(tr, 14)
    plus_di_raw = 100 * wilders_smoothing(plus_dm, 14) / atr_raw
    minus_di_raw = 100 * wilders_smoothing(minus_dm, 14) / atr_raw
    
    # DX and ADX
    dx_raw = 100 * np.abs(plus_di_raw - minus_di_raw) / (plus_di_raw + minus_di_raw)
    dx_raw = np.where((plus_di_raw + minus_di_raw) == 0, 0, dx_raw)
    adx = wilders_smoothing(dx_raw, 14)
    
    # Volume average (20-period) on 1d
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 6h Alligator to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # Align 1d ADX and volume MA to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Bullish alignment: price > Jaw AND Teeth > Lips
            bullish = price > jaw_val and teeth_val > lips_val
            # Bearish alignment: price < Jaw AND Teeth < Lips
            bearish = price < jaw_val and teeth_val < lips_val
            
            # Long: bullish AND ADX > 25 (trending) AND volume > 1.5x avg
            if bullish and adx_val > 25 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: bearish AND ADX > 25 (trending) AND volume > 1.5x avg
            elif bearish and adx_val > 25 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Teeth crosses below Jaw (Teeth < Jaw) OR ADX < 20 (range market)
            if teeth_val < jaw_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Teeth crosses above Jaw (Teeth > Jaw) OR ADX < 20 (range market)
            if teeth_val > jaw_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dADX_Volume_Filter"
timeframe = "6h"
leverage = 1.0