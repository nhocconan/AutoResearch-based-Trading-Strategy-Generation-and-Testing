#!/usr/bin/env python3
"""
Hypothesis: 1-day ADX with 1-week trend filter for trend-following in trending markets and mean-reversion in ranging markets.
- Long when ADX(14) > 25 (trending) and price > EMA(50) or ADX(14) < 20 (ranging) and price < Bollinger Lower Band(20,2)
- Short when ADX(14) > 25 (trending) and price < EMA(50) or ADX(14) < 20 (ranging) and price > Bollinger Upper Band(20,2)
- Uses 1-week EMA(50) for trend direction and 1-day Bollinger Bands for mean reversion.
- Designed to work in both bull (trend follow) and bear/range (mean revert) markets.
- Target: 20-60 trades over 4 years (5-15/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1-day data for ADX and Bollinger Bands - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1-week data for EMA trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-day ADX (14)
    plus_dm = np.zeros(len(df_1d))
    minus_dm = np.zeros(len(df_1d))
    tr = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        high_diff = df_1d['high'].iloc[i] - df_1d['high'].iloc[i-1]
        low_diff = df_1d['low'].iloc[i-1] - df_1d['low'].iloc[i]
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
        tr[i] = max(df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                    abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                    abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1]))
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Calculate 1-day Bollinger Bands (20,2)
    sma_20 = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Calculate 1-week EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False).mean().values
    
    # Align all indicators to 1-day timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Entry conditions
            if adx_aligned[i] > 25:  # Trending market
                # Trend following: go with 1-week EMA direction
                if close[i] > ema_50_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < ema_50_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:  # Ranging market (ADX < 25)
                # Mean reversion: fade Bollinger Bands
                if close[i] < bb_lower_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] > bb_upper_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: ADX drops below 20 (trend weakening) OR price crosses 1-week EMA
                if adx_aligned[i] < 20 or close[i] < ema_50_1w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: ADX drops below 20 OR price crosses 1-week EMA
                if adx_aligned[i] < 20 or close[i] > ema_50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_ADX_Trend_Range_Switch"
timeframe = "1d"
leverage = 1.0