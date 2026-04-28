#!/usr/bin/env python3
"""
6h_ADX_Strength_Trend_Filter_WeeklyPivot_Momentum
Hypothesis: Use ADX to identify strong trending regimes (>25) and trade pullbacks to weekly pivot support/resistance in the direction of the weekly trend. This combines trend strength with institutional reference levels to work in both bull and bear markets by only taking trades when momentum is aligned with the weekly structure. Target: 20-35 trades/year per symbol to minimize fee decay.
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
    
    # Get weekly data for trend and pivot levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate weekly ADX (trend strength)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # True Range
    tr1 = high_weekly[1:] - low_weekly[1:]
    tr2 = np.abs(high_weekly[1:] - close_weekly[:-1])
    tr3 = np.abs(low_weekly[1:] - close_weekly[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_weekly[1:] - high_weekly[:-1]) > (low_weekly[:-1] - low_weekly[1:]), 
                       np.maximum(high_weekly[1:] - high_weekly[:-1], 0), 0)
    dm_minus = np.where((low_weekly[:-1] - low_weekly[1:]) > (high_weekly[1:] - high_weekly[:-1]), 
                        np.maximum(low_weekly[:-1] - low_weekly[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period])  # skip first NaN
        for i in range(period, len(data)):
            if not np.isnan(data[i]) and not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_period = 14
    tr_atr = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # DI+ and DI-
    di_plus = np.where(tr_atr != 0, 100 * dm_plus_smooth / tr_atr, 0)
    di_minus = np.where(tr_atr != 0, 100 * dm_minus_smooth / tr_atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, atr_period)
    
    # Weekly pivot points (based on previous week)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    
    # Weekly support/resistance levels
    r1 = 2 * weekly_pivot - weekly_low
    s1 = 2 * weekly_pivot - weekly_high
    r2 = weekly_pivot + weekly_range
    s2 = weekly_pivot - weekly_range
    
    # Align weekly indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Determine weekly trend direction
    weekly_uptrend = weekly_close > weekly_pivot
    weekly_downtrend = weekly_close < weekly_pivot
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_weekly, weekly_uptrend.astype(float))
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_weekly, weekly_downtrend.astype(float))
    
    # 60-period EMA for 6x dynamic support/resistance
    ema_60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Volume filter: above average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(weekly_trend_up_aligned[i]) or np.isnan(weekly_trend_down_aligned[i]) or
            np.isnan(ema_60[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Strong trend filter (ADX > 25 indicates strong trend)
        strong_trend = adx_aligned[i] > 25
        
        # Price near weekly support/resistance (within 0.5% tolerance)
        near_r1 = np.abs(close[i] - r1_aligned[i]) / close[i] < 0.005
        near_s1 = np.abs(close[i] - s1_aligned[i]) / close[i] < 0.005
        near_r2 = np.abs(close[i] - r2_aligned[i]) / close[i] < 0.005
        near_s2 = np.abs(close[i] - s2_aligned[i]) / close[i] < 0.005
        
        # Pullback to weekly support in uptrend
        pullback_to_support = (near_s1 or near_s2) and weekly_trend_up_aligned[i] > 0.5
        
        # Pullback to weekly resistance in downtrend
        pullback_to_resistance = (near_r1 or near_r2) and weekly_trend_down_aligned[i] > 0.5
        
        # Additional confirmation: price above/below 60 EMA
        above_ema = close[i] > ema_60[i]
        below_ema = close[i] < ema_60[i]
        
        # Entry conditions
        long_entry = strong_trend and pullback_to_support and above_ema and volume_filter[i]
        short_entry = strong_trend and pullback_to_resistance and below_ema and volume_filter[i]
        
        # Exit conditions: trend weakening or opposite signal
        trend_weakening = adx_aligned[i] < 20
        opposite_signal = (pullback_to_resistance and weekly_trend_up_aligned[i] > 0.5) or \
                         (pullback_to_support and weekly_trend_down_aligned[i] > 0.5)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (trend_weakening or opposite_signal) and position != 0:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ADX_Strength_Trend_Filter_WeeklyPivot_Momentum"
timeframe = "6h"
leverage = 1.0