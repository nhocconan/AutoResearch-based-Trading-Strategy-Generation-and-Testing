#!/usr/bin/env python3
"""
12h_1d_Camarilla_Breakout_Enhanced
Hypothesis: Combines daily Camarilla pivot breakouts with 12h trend filter and volume confirmation.
Uses tighter breakout criteria (H4/L4 instead of H3/L3) and adds ADX trend strength filter to reduce false signals.
Designed for low-frequency, high-quality trades in both bull and bear markets by filtering for strong trends.
Target: 15-25 trades/year to minimize fee drag while capturing significant moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Breakout_Enhanced"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY CAMARILLA PIVOT CALCULATION ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    H3 = np.zeros(len(df_1d))
    L3 = np.zeros(len(df_1d))
    H4 = np.zeros(len(df_1d))
    L4 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        range_ = high_1d[i] - low_1d[i]
        if range_ <= 0:
            H3[i] = H4[i] = L3[i] = L4[i] = close_1d[i]
        else:
            H3[i] = close_1d[i] + range_ * 1.1 / 4
            L3[i] = close_1d[i] - range_ * 1.1 / 4
            H4[i] = close_1d[i] + range_ * 1.1 / 2
            L4[i] = close_1d[i] - range_ * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    H3_12h = align_htf_to_ltf(prices, df_1d, H3)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3)
    H4_12h = align_htf_to_ltf(prices, df_1d, H4)
    L4_12h = align_htf_to_ltf(prices, df_1d, L4)
    
    # === 12h TREND FILTER (ADX) ===
    # Calculate ADX for trend strength
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth the values
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[1:period+1])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    period = 14
    tr_smooth = smooth_wilder(tr, period)
    plus_di = 100 * smooth_wilder(plus_dm, period) / tr_smooth
    minus_di = 100 * smooth_wilder(minus_dm, period) / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_wilder(dx, period)
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(H4_12h[i]) or np.isnan(L4_12h[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with tighter criteria
        # Long: Price breaks above H4 with volume + strong trend (ADX > 25)
        long_breakout = (close[i] > H4_12h[i]) and (vol_ratio[i] > 2.0) and (adx[i] > 25)
        
        # Short: Price breaks below L4 with volume + strong trend (ADX > 25)
        short_breakout = (close[i] < L4_12h[i]) and (vol_ratio[i] > 2.0) and (adx[i] > 25)
        
        # Exit: Price returns to opposite H3/L3 level or trend weakens
        exit_long = (position == 1) and ((close[i] < L3_12h[i]) or (adx[i] < 20))
        exit_short = (position == -1) and ((close[i] > H3_12h[i]) or (adx[i] < 20))
        
        # Execute trades
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals