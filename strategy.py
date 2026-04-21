#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA34 (trend filter)
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Load weekly data for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA34 (trend filter)
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to 6h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 1d ATR (14-period) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_1d = np.zeros_like(tr)
    atr_1d[13] = np.mean(tr[1:14])
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align 1d ATR to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily pivot points (classic)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    
    # Handle first value
    high_prev[0] = high_1d[0]
    low_prev[0] = low_1d[0]
    close_prev[0] = close_1d[0]
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    r1 = 2 * pivot - low_prev
    s1 = 2 * pivot - high_prev
    r2 = pivot + (high_prev - low_prev)
    s2 = pivot - (high_prev - low_prev)
    r3 = high_prev + 2 * (pivot - low_prev)
    s3 = low_prev - 2 * (high_prev - pivot)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Skip if data not ready
        if (np.isnan(ema34_1d[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price (6h close)
        price_close = prices['close'].iloc[i]
        
        # Determine trend direction from daily and weekly EMA
        daily_trend_up = price_close > ema34_1d[i]
        weekly_trend_up = price_close > ema34_1w_aligned[i]
        
        # Volatility filter: only trade when volatility is above average
        vol_filter = atr_1d_aligned[i] > np.nanmedian(atr_1d_aligned[max(0, i-50):i+1]) if i >= 50 else True
        
        if position == 0:
            # Enter long: price above S1, daily and weekly trend up, volatility filter
            if (price_close > s1_aligned[i] and daily_trend_up and weekly_trend_up and vol_filter):
                signals[i] = 0.25
                position = 1
            # Enter short: price below R1, daily and weekly trend down, volatility filter
            elif (price_close < r1_aligned[i] and not daily_trend_up and not weekly_trend_up and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price below pivot or trend changes
                if (price_close < pivot_aligned[i] or not daily_trend_up):
                    exit_signal = True
            elif position == -1:
                # Exit short: price above pivot or trend changes
                if (price_close > pivot_aligned[i] or daily_trend_up):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_PivotTrend_EMA34"
timeframe = "6h"
leverage = 1.0