#!/usr/bin/env python3
# 1d_Weekly_Trend_Signal_v1
# Hypothesis: Use weekly (1w) ADX and price position relative to weekly EMA20 for trend strength and direction.
# Enter long when weekly ADX > 25 and price is above weekly EMA20, short when ADX > 25 and price below weekly EMA20.
# Use daily timeframe for execution with tight position sizing (0.25) to limit risk.
# Weekly timeframe filters noise, reducing whipsaws in sideways markets. Targets 15-25 trades/year.
# Works in both bull (trend following) and bear (short signals) markets.

name = "1d_Weekly_Trend_Signal_v1"
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
    
    # Get weekly data for ADX and EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX components (14-period)
    period = 14
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (EMA with alpha=1/period)
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period])
        # Subsequent values: EMA-style smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr_period = WilderSmoothing(tr, period)
    dm_plus_period = WilderSmoothing(dm_plus, period)
    dm_minus_period = WilderSmoothing(dm_minus, period)
    
    # Avoid division by zero
    dx = np.full_like(tr_period, np.nan)
    mask = tr_period > 0
    dx[mask] = 100 * np.abs(dm_plus_period[mask] - dm_minus_period[mask]) / tr_period[mask]
    
    # ADX is smoothed DX
    adx = WilderSmoothing(dx, period)
    
    # Weekly EMA20
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly indicators to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Warmup period
        # Skip if any critical value is NaN
        if np.isnan(adx_aligned[i]) or np.isnan(ema_20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: strong uptrend (ADX > 25) and price above weekly EMA20
            if adx_aligned[i] > 25 and close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: strong downtrend (ADX > 25) and price below weekly EMA20
            elif adx_aligned[i] > 25 and close[i] < ema_20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend weakens (ADX < 20) or price crosses below EMA20
            if adx_aligned[i] < 20 or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend weakens (ADX < 20) or price crosses above EMA20
            if adx_aligned[i] < 20 or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals