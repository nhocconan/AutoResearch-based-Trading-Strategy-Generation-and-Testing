#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d ADX trend filter + volume confirmation.
# Williams %R identifies overbought/oversold conditions for mean reversion entries.
# ADX from daily timeframe filters for trending vs ranging markets - only trade in strong trends (ADX > 25).
# Volume confirmation ensures momentum behind the move.
# Designed to work in both bull and bear markets by following the 1d ADX trend direction.
# Target: 20-35 trades/year per symbol to avoid excessive fee drift.
name = "6h_WilliamsR_1dADX_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smoothed_avg(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(x[1:period]) if np.any(~np.isnan(x[1:period])) else 0
            # Subsequent values are smoothed
            for i in range(period, len(x)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + x[i]) / period
                else:
                    result[i] = np.nan
        return result
    
    atr = smoothed_avg(tr, 14)
    dm_plus_smooth = smoothed_avg(dm_plus, 14)
    dm_minus_smooth = smoothed_avg(dm_minus, 14)
    
    # Directional Indicators
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smoothed_avg(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams %R (14-period) on 6h data
    def williams_r(high, low, close, period=14):
        highest_high = np.full_like(high, np.nan)
        lowest_low = np.full_like(low, np.nan)
        for i in range(len(high)):
            if i >= period - 1:
                start_idx = i - period + 1
                highest_high[i] = np.nanmax(high[start_idx:i+1])
                lowest_low[i] = np.nanmin(low[start_idx:i+1])
        wr = np.where((highest_high - lowest_low) != 0, 
                      -100 * (highest_high - close) / (highest_high - lowest_low), -50)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Volume confirmation: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema > 0, volume / vol_ema, 1.0) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for Williams %R and ADX
    
    for i in range(start_idx, n):
        if (np.isnan(wr[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) in strong trend
            long_condition = (wr[i] < -80) and strong_trend and vol_spike[i]
            # Short entry: Williams %R overbought (> -20) in strong trend
            short_condition = (wr[i] > -20) and strong_trend and vol_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Williams %R returns above -50 or trend weakens
            if (wr[i] > -50) or (not strong_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Williams %R returns below -50 or trend weakens
            if (wr[i] < -50) or (not strong_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals