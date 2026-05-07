#!/usr/bin/env python3
# 6h_WilliamsAlligator_1dTrend_WeeklyFilter
# Hypothesis: Combines Williams Alligator (13,8,5 SMAs) on 6h for trend direction, filtered by 1d EMA50 trend and weekly ADX > 25 for strong trends.
# Uses Williams Alligator crossover for entries (Jaws-Teeth-Lips) with trend and strength filters to work in both bull and bear markets.
# Designed for 6h to achieve 50-150 total trades over 4 years with low frequency and high conviction signals.

name = "6h_WilliamsAlligator_1dTrend_WeeklyFilter"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 6h: SMAs of median price
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    median_price = (high + low) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average (SMMA)"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Price) / Period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    smma_13 = smma(median_price, 13)
    smma_8 = smma(median_price, 8)
    smma_5 = smma(median_price, 5)
    
    # Shift SMMA according to Alligator specification
    jaws = np.roll(smma_13, 8)   # Jaw: 13-period SMMA shifted 8 bars
    teeth = np.roll(smma_8, 5)   # Teeth: 8-period SMMA shifted 5 bars
    lips = np.roll(smma_5, 3)    # Lips: 5-period SMMA shifted 3 bars
    
    # Align Alligator lines to 6h (already on 6h, but ensure proper alignment)
    # Since we calculated on 6h data directly, no alignment needed for Alligator
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly ADX for trend strength filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(arr, period):
        """Wilder's smoothing (equivalent to EMA with alpha=1/period)"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[:period])
        # Subsequent values: Wilder = (Prev * (Period-1) + Current) / Period
        for i in range(period, len(arr)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w > 0, dm_plus_smooth / atr_1w * 100, 0)
    di_minus = np.where(atr_1w > 0, dm_minus_smooth / atr_1w * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1w = wilders_smoothing(dx, 14)
    adx_1w_6h = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_6h[i]) or np.isnan(adx_1w_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator signals:
        # Lips above Teeth above Jaws = bullish alignment
        # Lips below Teeth below Jaws = bearish alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaws[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaws[i]
        
        # Trend filter: price vs 1d EMA50
        above_1d_ema = close[i] > ema_50_1d_6h[i]
        below_1d_ema = close[i] < ema_50_1d_6h[i]
        
        # Strength filter: weekly ADX > 25
        strong_trend = adx_1w_6h[i] > 25
        
        if position == 0:
            # Long: Bullish Alligator alignment + above 1d EMA50 + strong trend
            if bullish_alignment and above_1d_ema and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment + below 1d EMA50 + strong trend
            elif bearish_alignment and below_1d_ema and strong_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bearish Alligator alignment OR weak trend
            if bearish_alignment or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bullish Alligator alignment OR weak trend
            if bullish_alignment or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals