#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Williams Alligator combination
# ADX > 25 filters for trending markets, Williams Alligator (Jaw/Teeth/Lips) provides entry/exit signals
# Jaw (13-period SMMA shifted 8 bars), Teeth (8-period SMMA shifted 5 bars), Lips (5-period SMMA shifted 3 bars)
# Long: Lips > Teeth > Jaw (bullish alignment) + ADX > 25
# Short: Lips < Teeth < Jaw (bearish alignment) + ADX > 25
# Uses 1d timeframe for ADX and Alligator to reduce noise and avoid overtrading
# Designed to work in both bull and bear markets by only trading when strong trends exist
name = "6h_ADX_WilliamsAlligator_1d"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for ADX and Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smooth TR, DM+, DM- using Wilder's smoothing (EMA-like)
        def WilderSmooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nansum(data[:period]) / period
            # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = result[i-1] - (result[i-1]/period) + data[i]
            return result
        
        atr = WilderSmooth(tr, period)
        dm_plus_smooth = WilderSmooth(dm_plus, period)
        dm_minus_smooth = WilderSmooth(dm_minus, period)
        
        # Avoid division by zero
        dm_plus_div = np.where(atr != 0, dm_plus_smooth / atr, 0)
        dm_minus_div = np.where(atr != 0, dm_minus_smooth / atr, 0)
        
        dx = np.where((dm_plus_div + dm_minus_div) != 0, 
                      np.abs(dm_plus_div - dm_minus_div) / (dm_plus_div + dm_minus_div) * 100, 0)
        
        # ADX is smoothed DX
        adx = WilderSmooth(dx, period)
        return adx
    
    # Calculate Williams Alligator (SMMA with specific shifts)
    def Smma(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period]) / period
        # Subsequent values: smoothed = (prev_smoothed * (period-1) + current) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    median_price_1d = (high_1d + low_1d) / 2
    jaw_raw = Smma(median_price_1d, 13)
    jaw = np.roll(jaw_raw, 8)  # Shift 8 bars forward
    
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    teeth_raw = Smma(median_price_1d, 8)
    teeth = np.roll(teeth_raw, 5)  # Shift 5 bars forward
    
    # Lips: 5-period SMMA of median price, shifted 3 bars
    lips_raw = Smma(median_price_1d, 5)
    lips = np.roll(lips_raw, 3)  # Shift 3 bars forward
    
    # Calculate ADX
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align to 6h timeframe
    jaw_6h = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_6h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_6h = align_htf_to_ltf(prices, df_1d, lips)
    adx_6h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or np.isnan(lips_6h[i]) or
            np.isnan(adx_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator conditions
        bullish_alignment = lips_6h[i] > teeth_6h[i] and teeth_6h[i] > jaw_6h[i]
        bearish_alignment = lips_6h[i] < teeth_6h[i] and teeth_6h[i] < jaw_6h[i]
        
        # ADX trend filter
        strong_trend = adx_6h[i] > 25
        
        if position == 0:
            # Long: bullish alignment + strong trend
            if bullish_alignment and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment + strong trend
            elif bearish_alignment and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish alignment or weak trend
            if bearish_alignment or adx_6h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish alignment or weak trend
            if bullish_alignment or adx_6h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals