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
    
    # Weekly EMA50 trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    r3 = pivot + (range_hl * 1.1)
    s3 = pivot - (range_hl * 1.1)
    r4 = pivot + (range_hl * 1.5)
    s4 = pivot - (range_hl * 1.5)
    
    # Shift to use previous day's pivots (avoid look-ahead)
    r3_prev = np.roll(r3, 1)
    s3_prev = np.roll(s3, 1)
    r4_prev = np.roll(r4, 1)
    s4_prev = np.roll(s4, 1)
    r3_prev[0] = np.nan
    s3_prev[0] = np.nan
    r4_prev[0] = np.nan
    s4_prev[0] = np.nan
    
    # Align daily Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3_prev)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3_prev)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4_prev)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4_prev)
    
    # Volume confirmation: current volume > 1.5 * 12-period average (6h * 12 = 3 days)
    volume_ma12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
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
    
    start_idx = 20  # Need R4/S4 and ATR MA10
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma12[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma10[i]) or 
            np.isnan(r4_6h[i]) or 
            np.isnan(s4_6h[i]) or
            np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 12-period average
        volume_filter = volume[i] > (1.5 * volume_ma12[i])
        # Volatility filter: ATR > ATR MA10 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma10[i]
        # Weekly trend filter
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R4 in weekly uptrend with volume and volatility
            if close[i] > r4_6h[i] and weekly_uptrend and volume_filter and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 in weekly downtrend with volume and volatility
            elif close[i] < s4_6h[i] and weekly_downtrend and volume_filter and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below R3 or volatility drops
            if close[i] < r3_6h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S3 or volatility drops
            if close[i] > s3_6h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R4_S4_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0