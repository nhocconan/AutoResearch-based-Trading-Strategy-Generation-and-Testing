#!/usr/bin/env python3
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
    
    # Get weekly data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly ATR(20) for volatility filter
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_20_1w = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_20_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_20_1w)
    
    # Calculate weekly EMA(20) for trend direction
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily pivot points from previous day
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points (classic formula)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Calculate 60-period EMA for 6h trend filter
    ema_60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(atr_20_1w_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(ema_60[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Volatility filter: avoid extremely high volatility periods
        vol_threshold = np.nanpercentile(atr_20_1w_aligned[max(0, i-100):i+1], 80) if i >= 50 else atr_20_1w_aligned[i]
        normal_volatility = atr_20_1w_aligned[i] < vol_threshold
        
        # Pivot-based signals
        at_pivot_support = abs(close[i] - s1_aligned[i]) < (pivot_aligned[i] - s1_aligned[i]) * 0.1
        at_pivot_resistance = abs(close[i] - r1_aligned[i]) < (r1_aligned[i] - pivot_aligned[i]) * 0.1
        breakout_above_r2 = close[i] > r2_aligned[i]
        breakout_below_s2 = close[i] < s2_aligned[i]
        
        # Long conditions: bounce at support in uptrend or breakout above resistance
        long_condition = (
            (weekly_uptrend and at_pivot_support and normal_volatility) or
            (breakout_above_r2 and weekly_uptrend and normal_volatility)
        )
        
        # Short conditions: bounce at resistance in downtrend or breakdown below support
        short_condition = (
            (weekly_downtrend and at_pivot_resistance and normal_volatility) or
            (breakout_below_s2 and weekly_downtrend and normal_volatility)
        )
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or volatility spike
        elif position == 1 and (not weekly_uptrend or not normal_volatility):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not weekly_downtrend or not normal_volatility):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyTrend_PivotBounce_VolatilityFilter"
timeframe = "6h"
leverage = 1.0