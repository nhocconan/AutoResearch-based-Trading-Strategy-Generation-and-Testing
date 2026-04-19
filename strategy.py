#!/usr/bin/env python3
"""
1d_Williams_Alligator_1wADX_Volume
Hypothesis: Williams Alligator identifies trend direction on daily timeframe, filtered by weekly ADX for trend strength and volume confirmation. Works in bull/bear via ADX filter and avoids false signals in chop.
"""

name = "1d_Williams_Alligator_1wADX_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: SMMA(13,8), SMMA(8,5), SMMA(5,3)
    def smoothed_ma(data, period):
        # Smoothed Moving Average (SMMA) using Wilder's smoothing (alpha = 1/period)
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] + alpha * (data[i] - result[i-1])
                else:
                    result[i] = np.nan
        return result
    
    # Daily data for Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    jaw = smoothed_ma(df_1d['close'].values, 13)  # Blue line (13-period)
    teeth = smoothed_ma(df_1d['close'].values, 8)  # Red line (8-period)
    lips = smoothed_ma(df_1d['close'].values, 5)   # Green line (5-period)
    
    # Align Alligator lines to daily timeframe (no shift needed as we use previous day's values)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Weekly ADX for trend strength filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Wilder's smoothing
        def WilderSmooth(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            if len(data) >= period:
                result[period-1] = np.nanmean(data[:period])
                for i in range(period, len(data)):
                    if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                        result[i] = result[i-1] + alpha * (data[i] - result[i-1])
                    else:
                        result[i] = np.nan
            return result
        
        atr = WilderSmooth(tr, period)
        dm_plus_smooth = WilderSmooth(dm_plus, period)
        dm_minus_smooth = WilderSmooth(dm_minus, period)
        
        # Avoid division by zero
        dx = np.full_like(close, np.nan)
        mask = (atr > 0) & ~np.isnan(atr) & ~np.isnan(dm_plus_smooth) & ~np.isnan(dm_minus_smooth)
        dx[mask] = 100 * np.abs(dm_plus_smooth[mask] - dm_minus_smooth[mask]) / (dm_plus_smooth[mask] + dm_minus_smooth[mask])
        
        adx = WilderSmooth(dx, period)
        return adx
    
    adx_1w = calculate_adx(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # ADX filter: only trade when ADX > 25 (trending market)
        strong_trend = adx_1w_aligned[i] > 25
        
        # Alligator alignment: lips > teeth > jaw = uptrend, lips < teeth < jaw = downtrend
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i] and 
                            teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i] and 
                            teeth_aligned[i] < jaw_aligned[i])
        
        if position == 0:
            # Long: bullish alignment + volume + strong trend
            if (bullish_alignment and 
                volume_confirm[i] and 
                strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment + volume + strong trend
            elif (bearish_alignment and 
                  volume_confirm[i] and 
                  strong_trend):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if bearish alignment or trend weakens (ADX < 20)
            if (bearish_alignment) or (adx_1w_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if bullish alignment or trend weakens (ADX < 20)
            if (bullish_alignment) or (adx_1w_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals