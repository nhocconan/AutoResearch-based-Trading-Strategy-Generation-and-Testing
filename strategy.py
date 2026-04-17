#!/usr/bin/env python3
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
    
    # Get weekly data for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily pivot points for entry levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Shift to use previous day's pivots
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    r2_prev = np.roll(r2, 1)
    s2_prev = np.roll(s2, 1)
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    r2_prev[0] = np.nan
    s2_prev[0] = np.nan
    
    # Align daily pivot levels to daily timeframe (same timeframe)
    r1_d = align_htf_to_ltf(prices, df_1d, r1_prev)
    s1_d = align_htf_to_ltf(prices, df_1d, s1_prev)
    r2_d = align_htf_to_ltf(prices, df_1d, r2_prev)
    s2_d = align_htf_to_ltf(prices, df_1d, s2_prev)
    
    # Volume confirmation: current volume > 1.8 * 20-day average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility filter and position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(r2_d[i]) or 
            np.isnan(s2_d[i]) or
            np.isnan(r1_d[i]) or 
            np.isnan(s1_d[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-day average
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA34
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R2 with volume and uptrend
            if close[i] > r2_d[i] and volume_filter and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume and downtrend
            elif close[i] < s2_d[i] and volume_filter and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below R1 or trend changes
            if close[i] < r1_d[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S1 or trend changes
            if close[i] > s1_d[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Pivot_R2_S2_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0