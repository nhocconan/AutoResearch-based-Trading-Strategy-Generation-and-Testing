#!/usr/bin/env python3
"""
1D_WILLIAMS_ALLIGATOR_1W_TREND_FILTER
Hypothesis: Williams Alligator (Jaws, Teeth, Lips) on 1-day timeframe confirms trend.
Enter long when Lips > Teeth > Jaws (bullish alignment) and price above Teeth.
Enter short when Lips < Teeth < Jaws (bearish alignment) and price below Teeth.
Use 1-week ADX > 25 as trend filter to avoid choppy markets.
Designed for ~15-25 trades/year on 1d to minimize fee drag and capture major trends.
Works in bull markets via long trends and bear markets via short trends.
"""
name = "1D_WILLIAMS_ALLIGATOR_1W_TREND_FILTER"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Williams Alligator on daily data
    # Jaws: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full(len(arr), np.nan)
        sma = np.nansum(arr[:period]) / period
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate SMMA for different periods
    jaws_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply forward shifts (Jaws +8, Teeth +5, Lips +3)
    jaws = np.full_like(jaws_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    for i in range(len(jaws)):
        if i + 8 < len(jaws) and not np.isnan(jaws_raw[i]):
            jaws[i + 8] = jaws_raw[i]
        if i + 5 < len(teeth) and not np.isnan(teeth_raw[i]):
            teeth[i + 5] = teeth_raw[i]
        if i + 3 < len(lips) and not np.isnan(lips_raw[i]):
            lips[i + 3] = lips_raw[i]
    
    # Get 1-week data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ADX on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    def WilderSmooth(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full(len(arr), np.nan)
        # First value is simple average
        result[period-1] = np.nansum(arr[:period]) / period
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = WilderSmooth(tr, 14)
    dm_plus_smooth = WilderSmooth(dm_plus, 14)
    dm_minus_smooth = WilderSmooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, dm_plus_smooth / atr * 100, 0)
    di_minus = np.where(atr != 0, dm_minus_smooth / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = WilderSmooth(dx, 14)
    
    # Align Alligator and ADX to 1d timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1w, jaws, additional_delay_bars=0)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth, additional_delay_bars=0)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips, additional_delay_bars=0)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # LONG: Bullish Alligator alignment (Lips > Teeth > Jaws) and price above Teeth
            if (lips_aligned[i] > teeth_aligned[i] > jaws_aligned[i] and 
                close[i] > teeth_aligned[i] and strong_trend):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish Alligator alignment (Lips < Teeth < Jaws) and price below Teeth
            elif (lips_aligned[i] < teeth_aligned[i] < jaws_aligned[i] and 
                  close[i] < teeth_aligned[i] and strong_trend):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator alignment breaks down (Lips <= Teeth) or ADX weakens
            if (lips_aligned[i] <= teeth_aligned[i] or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator alignment breaks down (Lips >= Teeth) or ADX weakens
            if (lips_aligned[i] >= teeth_aligned[i] or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals