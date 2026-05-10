#!/usr/bin/env python3
# 4h_RVI_Crossover_ADX_TrendFilter
# Hypothesis: RVI (Relative Vigor Index) crossovers filtered by ADX trend strength and volume surge.
# In trending markets (ADX > 25), RVI crossovers provide high-probability entries.
# Volume surge confirms institutional participation. Works in both bull and bear trends.
# Target: 20-40 trades/year to minimize fee drag.

name = "4h_RVI_Crossover_ADX_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Wilder smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilders_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4h price data for RVI calculation
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RVI calculation (10-period)
    # Numerator: (close - open) + 2*(close_prev - open_prev) + 2*(close_prev2 - open_prev2) + (close_prev3 - open_prev3)
    # Denominator: (high - low) + 2*(high_prev - low_prev) + 2*(high_prev2 - low_prev2) + (high_prev3 - low_prev3)
    num = (close - open_price) + \
          2 * np.roll(close - open_price, 1) + \
          2 * np.roll(close - open_price, 2) + \
          np.roll(close - open_price, 3)
    den = (high - low) + \
          2 * np.roll(high - low, 1) + \
          2 * np.roll(high - low, 2) + \
          np.roll(high - low, 3)
    
    # Handle first 3 values
    num[:3] = np.nan
    den[:3] = np.nan
    
    # Avoid division by zero
    rvi_raw = np.where(den != 0, num / den, 0)
    
    # Smooth RVI with SMA (4-period)
    rvi = pd.Series(rvi_raw).rolling(window=4, min_periods=4).mean().values
    rvi_signal = pd.Series(rvi).rolling(window=4, min_periods=4).mean().values  # Signal line
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RVI (10+4+4=18), ADX (14+14=28), volume MA (20)
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rvi[i]) or 
            np.isnan(rvi_signal[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        # RVI crossover signals
        rvi_cross_above = rvi[i] > rvi_signal[i] and rvi[i-1] <= rvi_signal[i-1]
        rvi_cross_below = rvi[i] < rvi_signal[i] and rvi[i-1] >= rvi_signal[i-1]
        
        if position == 0:
            # Long: RVI bullish crossover with strong trend and volume surge
            if rvi_cross_above and strong_trend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: RVI bearish crossover with strong trend and volume surge
            elif rvi_cross_below and strong_trend and volume_surge:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RVI bearish crossover OR trend weakening
            if rvi_cross_below or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RVI bullish crossover OR trend weakening
            if rvi_cross_above or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals