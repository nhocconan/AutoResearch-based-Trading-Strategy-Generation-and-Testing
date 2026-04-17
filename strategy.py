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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    r3 = high_1w + 2 * (pivot - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot)
    
    # Shift to use previous week's pivots
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    r2_prev = np.roll(r2, 1)
    s2_prev = np.roll(s2, 1)
    r3_prev = np.roll(r3, 1)
    s3_prev = np.roll(s3, 1)
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    r2_prev[0] = np.nan
    s2_prev[0] = np.nan
    r3_prev[0] = np.nan
    s3_prev[0] = np.nan
    
    # Align weekly pivot levels to daily timeframe
    r1_daily = align_htf_to_ltf(prices, df_1w, r1_prev)
    s1_daily = align_htf_to_ltf(prices, df_1w, s1_prev)
    r2_daily = align_htf_to_ltf(prices, df_1w, r2_prev)
    s2_daily = align_htf_to_ltf(prices, df_1w, s2_prev)
    r3_daily = align_htf_to_ltf(prices, df_1w, r3_prev)
    s3_daily = align_htf_to_ltf(prices, df_1w, s3_prev)
    
    # Volume confirmation: current volume > 1.5 * 5-period average
    volume_ma5 = pd.Series(volume).rolling(window=5, min_periods=5).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma10 = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 14  # Need ATR and ATR MA10
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma5[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma10[i]) or 
            np.isnan(r1_daily[i]) or 
            np.isnan(s1_daily[i]) or
            np.isnan(r3_daily[i]) or 
            np.isnan(s3_daily[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 5-period average
        volume_filter = volume[i] > (1.5 * volume_ma5[i])
        # Volatility filter: ATR > ATR MA10 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma10[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume and volatility (strong breakout)
            if close[i] > r3_daily[i] and volume_filter and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume and volatility (strong breakdown)
            elif close[i] < s3_daily[i] and volume_filter and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below R2 or volatility drops
            if close[i] < r2_daily[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S2 or volatility drops
            if close[i] > s2_daily[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_R3_S3_Breakout_Vol"
timeframe = "1d"
leverage = 1.0