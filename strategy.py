#!/usr/bin/env python3
"""
6h_1d_WeeklyPivot_Trend_Follow
Hypothesis: Trend-following strategy using weekly pivot points (from weekly data) to establish long-term trend direction, with 1d ATR volatility filter and 60-minute EMA for entry timing on 6h timeframe. Weekly pivots provide robust structural levels that work in both bull and bear markets by identifying key support/resistance from higher timeframe. The 60-minute EMA (aligned to 6h) filters for entries in the direction of the short-term trend, while the 1d ATR filter avoids choppy markets. Designed for low trade frequency (15-30 trades/year) with discrete position sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points (standard floor pivot)"""
    pivot = (high + low + close) / 3.0
    R1 = 2 * pivot - low
    S1 = 2 * pivot - high
    R2 = pivot + (high - low)
    S2 = pivot - (high - low)
    return pivot, R1, R2, S1, S2

def calculate_ema(values, period):
    """Calculate Exponential Moving Average with proper handling of initial period"""
    if len(values) < period:
        return np.full(len(values), np.nan)
    
    ema = np.zeros(len(values))
    multiplier = 2.0 / (period + 1)
    
    # Initialize first value as simple average
    ema[0] = values[0]
    
    for i in range(1, len(values)):
        ema[i] = (values[i] - ema[i-1]) * multiplier + ema[i-1]
    
    # Set first (period-1) values to NaN to ensure proper warmup
    ema[:period-1] = np.nan
    return ema

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr = np.zeros_like(tr)
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for pivot points (trend filter)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Load daily data once for ATR (volatility filter)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    weekly_pivot = np.zeros(len(df_weekly))
    weekly_R1 = np.zeros(len(df_weekly))
    weekly_S1 = np.zeros(len(df_weekly))
    
    for i in range(len(df_weekly)):
        pivot, R1, _, S1, _ = calculate_weekly_pivot(high_weekly[i], low_weekly[i], close_weekly[i])
        weekly_pivot[i] = pivot
        weekly_R1[i] = R1
        weekly_S1[i] = S1
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_R1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_R1)
    weekly_S1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_S1)
    
    # Calculate daily ATR for volatility filter
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    atr_daily = calculate_atr(high_daily, low_daily, close_daily, 14)
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    
    # Calculate 60-period EMA on 6h close for entry timing (using 60 periods because 60*6min=6h)
    close_6h = prices['close'].values
    ema_60 = calculate_ema(close_6h, 60)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after EMA warmup
        # Skip if indicators not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_R1_aligned[i]) or 
            np.isnan(weekly_S1_aligned[i]) or np.isnan(atr_daily_aligned[i]) or 
            np.isnan(ema_60[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Volatility filter: avoid extremely low volatility (choppy markets)
        # Only trade when ATR is above its 30th percentile (adaptive threshold)
        if i >= 30:
            atr_threshold = np.percentile(atr_daily_aligned[max(0, i-29):i+1], 30)
            vol_filter = atr_daily_aligned[i] > atr_threshold
        else:
            vol_filter = True
        
        if position == 0:
            # Long: price above weekly pivot AND above 60-EMA (uptrend)
            if price > weekly_pivot_aligned[i] and price > ema_60[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot AND below 60-EMA (downtrend)
            elif price < weekly_pivot_aligned[i] and price < ema_60[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly pivot OR below 60-EMA
            if price < weekly_pivot_aligned[i] or price < ema_60[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly pivot OR above 60-EMA
            if price > weekly_pivot_aligned[i] or price > ema_60[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_WeeklyPivot_Trend_Follow"
timeframe = "6h"
leverage = 1.0