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
    
    # Get daily data for pivot calculation (weekly pivot from daily data)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points using last 5 daily bars (weekly approximation)
    # Use 5-day high, low, close for weekly pivot
    window = 5
    weekly_high = pd.Series(high_1d).rolling(window=window, min_periods=window).max().values
    weekly_low = pd.Series(low_1d).rolling(window=window, min_periods=window).min().values
    weekly_close = pd.Series(close_1d).rolling(window=window, min_periods=window).last().values
    
    pivot_weekly = (weekly_high + weekly_low + weekly_close) / 3.0
    r1_weekly = 2 * pivot_weekly - weekly_low
    s1_weekly = 2 * pivot_weekly - weekly_high
    r2_weekly = pivot_weekly + (weekly_high - weekly_low)
    s2_weekly = pivot_weekly - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to daily timeframe
    pivot_weekly_aligned = align_htf_to_ltf(prices, df_1d, pivot_weekly)
    r1_weekly_aligned = align_htf_to_ltf(prices, df_1d, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_1d, s1_weekly)
    r2_weekly_aligned = align_htf_to_ltf(prices, df_1d, r2_weekly)
    s2_weekly_aligned = align_htf_to_ltf(prices, df_1d, s2_weekly)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily ATR for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_weekly_aligned[i]) or np.isnan(r1_weekly_aligned[i]) or np.isnan(s1_weekly_aligned[i]) or
            np.isnan(r2_weekly_aligned[i]) or np.isnan(s2_weekly_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above weekly EMA50 for long, below for short
        long_trend = close[i] > ema50_1w_aligned[i]
        short_trend = close[i] < ema50_1w_aligned[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr[i] > np.nanpercentile(atr[max(0, i-100):i+1], 20) if i >= 100 else True
        
        if position == 0:
            # Long: price breaks above S1 with trend alignment
            if long_trend and vol_filter and close[i] > s1_weekly_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below R1 with trend alignment
            elif short_trend and vol_filter and close[i] < r1_weekly_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below pivot or reverses at R2
            if close[i] < pivot_weekly_aligned[i] or close[i] > r2_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above pivot or reverses at S2
            if close[i] > pivot_weekly_aligned[i] or close[i] < s2_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_S1R1_EMA50Trend"
timeframe = "1d"
leverage = 1.0