#!/usr/bin/env python3
"""
6h_ADX_Alligator_1dTrend_Filter
Hypothesis: On 6h timeframe, combine ADX (>25) for trend strength with Williams Alligator (jaw/teeth/lips) for entry timing, filtered by 1d EMA50 trend. ADX ensures we only trade in strong trends, reducing whipsaw in ranging markets. Alligator provides entry signals when lips cross teeth in direction of trend, with lips crossing jaw confirming acceleration. 1d trend filter ensures alignment with higher timeframe momentum. Designed for 50-150 total trades over 4 years to minimize fee drag. Works in both bull and bear markets via trend filter and ADX requirement.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ADX on 6h (primary timeframe)
    # ADX calculation requires +DM, -DM, TR
    # +DM = high[t] - high[t-1] if high[t] - high[t-1] > low[t-1] - low[t] and > 0, else 0
    # -DM = low[t-1] - low[t] if low[t-1] - low[t] > high[t] - high[t-1] and > 0, else 0
    # TR = max(high[t] - low[t], abs(high[t] - close[t-1]), abs(low[t] - close[t-1]))
    # +DM_smooth = smoothed +DM (Wilder's smoothing)
    # -DM_smooth = smoothed -DM
    # TR_smooth = smoothed TR
    # +DI = 100 * +DM_smooth / TR_smooth
    # -DI = 100 * -DM_smooth / TR_smooth
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = smoothed DX
    
    # Calculate +DM and -DM
    high_diff = np.diff(high, prepend=high[0])
    low_diff = np.diff(low, prepend=low[0])  # low[t] - low[t-1]
    
    plus_dm = np.where((high_diff > -low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((-low_diff > high_diff) & (-low_diff > 0), -low_diff, 0.0)
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    tr_smooth = wilders_smoothing(tr, period)
    
    # Avoid division by zero
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smoothing(dx, period)
    
    # Calculate Williams Alligator on 6h (using SMMA - smoothed moving average)
    # Jaw: SMMA(median price, 13, 8)
    # Teeth: SMMA(median price, 8, 5)
    # Lips: SMMA(median price, 5, 3)
    # median price = (high + low) / 2
    median_price = (high + low) / 2
    
    def smma(data, period):
        # Smoothed Moving Average: similar to EMA but with alpha = 1/period
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period - 1) + data[i]) / period
        return result
    
    jaw_period, jaw_shift = 13, 8
    teeth_period, teeth_shift = 8, 5
    lips_period, lips_shift = 5, 3
    
    jaw = smma(median_price, jaw_period)
    teeth = smma(median_price, teeth_period)
    lips = smma(median_price, lips_period)
    
    # Apply shifts (Alligator lines are shifted into the future)
    jaw = np.concatenate([np.full(jaw_shift, np.nan), jaw[:-jaw_shift]]) if jaw_shift > 0 else jaw
    teeth = np.concatenate([np.full(teeth_shift, np.nan), teeth[:-teeth_shift]]) if teeth_shift > 0 else teeth
    lips = np.concatenate([np.full(lips_shift, np.nan), lips[:-lips_shift]]) if lips_shift > 0 else lips
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for all calculations
    start_idx = max(100, 50)  # ADX, Alligator, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        adx_val = adx[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_50_val = ema_50_aligned[i]
        close_val = close[i]
        
        # Conditions
        strong_trend = adx_val > 25
        uptrend_1d = close_val > ema_50_val  # Using 6h close vs 1d EMA (aligned)
        downtrend_1d = close_val < ema_50_val
        
        # Alligator signals
        # Lips crossing above teeth = potential long
        lips_above_teeth = lips_val > teeth_val
        # Lips crossing below teeth = potential short
        lips_below_teeth = lips_val < teeth_val
        # Lips crossing above jaw = acceleration confirmation
        lips_above_jaw = lips_val > jaw_val
        # Lips crossing below jaw = acceleration confirmation
        lips_below_jaw = lips_val < jaw_val
        
        if position == 0:
            # Look for entry signals
            # Long: strong trend + uptrend 1d + lips cross above teeth + lips above jaw
            long_signal = strong_trend and uptrend_1d and lips_above_teeth and lips_above_jaw
            # Short: strong trend + downtrend 1d + lips cross below teeth + lips below jaw
            short_signal = strong_trend and downtrend_1d and lips_below_teeth and lips_below_jaw
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Trend weakens: ADX < 20
            # 2. Trend reverses: close crosses below 1d EMA
            # 3. Alligator reverses: lips cross below teeth
            if adx_val < 20 or not uptrend_1d or lips_val < teeth_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Trend weakens: ADX < 20
            # 2. Trend reverses: close crosses above 1d EMA
            # 3. Alligator reverses: lips cross above teeth
            if adx_val < 20 or not downtrend_1d or lips_val > teeth_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_Alligator_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0