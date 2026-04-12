#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with daily pivot-based mean reversion and weekly trend filter
# Uses daily pivot points (PP, R1, S1) for mean reversion entries and weekly ADX for trend filtering
# In ranging markets (ADX < 25): fade at R1/S1 with rejection candles
# In trending markets (ADX >= 25): breakout continuation at R2/S2
# Designed for low trade frequency (20-40/year) to minimize fee drag on 6h chart
# Works in both bull and bear markets via adaptive regime filtering

name = "6h_1d_1w_pivot_adaptive"
timeframe = "6h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    # Pivot point and support/resistance levels
    pp = (high_prev + low_prev + close_prev) / 3
    r1 = 2 * pp - low_prev
    s1 = 2 * pp - high_prev
    r2 = pp + (high_prev - low_prev)
    s2 = pp - (high_prev - low_prev)
    
    # Align daily pivots to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly ADX
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_1w_aligned[i]
        is_trending = adx >= 25
        
        if is_trending:
            # Trending market: breakout continuation
            # Long: break above R2 with close > R1
            if close[i] > r2_aligned[i] and close[i-1] <= r2_aligned[i-1] and close[i] > r1_aligned[i]:
                if position != 1:
                    position = 1
                    signals[i] = 0.25
            # Short: break below S2 with close < S1
            elif close[i] < s2_aligned[i] and close[i-1] >= s2_aligned[i-1] and close[i] < s1_aligned[i]:
                if position != -1:
                    position = -1
                    signals[i] = -0.25
            else:
                # Hold current position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            # Ranging market: mean reversion at R1/S1
            # Long: rejection at S1 (close > S1 and open <= S1)
            if close[i] > s1_aligned[i] and prices['open'].values[i] <= s1_aligned[i]:
                if position != 1:
                    position = 1
                    signals[i] = 0.25
            # Short: rejection at R1 (close < R1 and open >= R1)
            elif close[i] < r1_aligned[i] and prices['open'].values[i] >= r1_aligned[i]:
                if position != -1:
                    position = -1
                    signals[i] = -0.25
            else:
                # Hold current position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
    
    return signals