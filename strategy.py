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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Shift to use previous day's pivots (avoid look-ahead)
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    
    # Align daily pivot levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_prev)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_prev)
    
    # Volume confirmation: current volume > 1.5 * 2-period average (12h * 2 = 24h)
    volume_ma2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma5 = pd.Series(atr).rolling(window=5, min_periods=5).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need EMA50 and ATR MA5
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma2[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma5[i]) or 
            np.isnan(r1_12h[i]) or 
            np.isnan(s1_12h[i]) or
            np.isnan(ema50_1w[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 2-period average
        volume_filter = volume[i] > (1.5 * volume_ma2[i])
        # Volatility filter: ATR > ATR MA5 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma5[i]
        # Weekly trend filter: price above/below weekly EMA50
        trend_up = close[i] > ema50_1w[i]
        trend_down = close[i] < ema50_1w[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume, volatility and uptrend
            if close[i] > r1_12h[i] and volume_filter and volatility_filter and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume, volatility and downtrend
            elif close[i] < s1_12h[i] and volume_filter and volatility_filter and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below R1 or volatility drops
            if close[i] < r1_12h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S1 or volatility drops
            if close[i] > s1_12h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0