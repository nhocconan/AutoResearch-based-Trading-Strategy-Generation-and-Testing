#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX Regime Filter
# Uses 12h ADX to determine market regime:
#   - ADX > 25 = trending market → Elder Ray signals valid
#   - ADX < 20 = ranging market → avoid trading
# Combines with 6h Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) for entries
# Designed for 6h timeframe targeting 12-37 trades/year with discrete sizing (0.25)
# Works in bull markets (strong Bull Power in uptrend) and bear markets (strong Bear Power in downtrend)
# ADX regime filter prevents whipsaws in choppy markets, a common failure point for oscillators

name = "6h_ElderRay_12hADX_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 12h data for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 12h ADX (Average Directional Index)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with indices
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(arr, period):
        """Wilder's smoothing (equivalent to EMA with alpha=1/period)"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(arr[:period])
        # Wilder's smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(arr)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    period_adx = 14
    tr_smooth = wilders_smoothing(tr, period_adx)
    dm_plus_smooth = wilders_smoothing(dm_plus, period_adx)
    dm_minus_smooth = wilders_smoothing(dm_minus, period_adx)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smoothing(dx, period_adx)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Get 6h data for Elder Ray
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h EMA(13) for Elder Ray
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    ema_13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_aligned = align_htf_to_ltf(prices, df_6h, ema_13)
    
    # Calculate Elder Ray components
    bull_power = high_6h - ema_13  # High - EMA13
    bear_power = ema_13 - low_6h   # EMA13 - Low
    
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_13_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: ADX > 25 = trending market (trade), ADX < 20 = ranging (avoid)
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] < 20
        
        if position == 0:
            # Enter long: strong Bull Power in trending market
            if is_trending and bull_power_aligned[i] > 0 and bear_power_aligned[i] < bull_power_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: strong Bear Power in trending market
            elif is_trending and bear_power_aligned[i] > 0 and bull_power_aligned[i] < bear_power_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power becomes stronger than Bull Power OR market turns ranging
            if bear_power_aligned[i] >= bull_power_aligned[i] or is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power becomes stronger than Bear Power OR market turns ranging
            if bull_power_aligned[i] >= bear_power_aligned[i] or is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals