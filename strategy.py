#!/usr/bin/env python3
"""
4h_12h_Camarilla_Breakout_Volume
Hypothesis: Trade Camarilla pivot breakouts (H4/L4) on 4h with volume confirmation and 12h trend filter (EMA25 > EMA50). 
Uses 12h EMA cross for trend direction to avoid whipsaws in ranging markets. Designed for 20-40 trades/year with clear breakout logic.
Works in bull (breakouts continue trend) and bear (failed reversals at pivot levels) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12H DATA FOR TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # EMA25 and EMA50 for trend
    ema25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Trend: EMA25 > EMA50 = uptrend, EMA25 < EMA50 = downtrend
    trend_12h = ema25_12h - ema50_12h
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === 4H INDICATORS: CAMARILLA PIVOTS FROM PREVIOUS DAY ===
    # Need daily high/low/close from previous day
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # H4 = Close + 1.1*(High-Low)*1.1/2
    # L4 = Close - 1.1*(High-Low)*1.1/2
    camarilla_width = 1.1 * (high_1d - low_1d) * 1.1 / 2
    h4 = close_1d + camarilla_width
    l4 = close_1d - camarilla_width
    
    # Align daily Camarilla levels to 4h (constant throughout the day)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # === 4H VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(trend_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume strength
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Long: price breaks above H4 with volume and uptrend
        long_signal = (close[i] > h4_aligned[i] and 
                      strong_volume and 
                      trend_12h_aligned[i] > 0)
        
        # Short: price breaks below L4 with volume and downtrend
        short_signal = (close[i] < l4_aligned[i] and 
                       strong_volume and 
                       trend_12h_aligned[i] < 0)
        
        # Exit: price returns to opposite level or trend reverses
        exit_long = (position == 1 and 
                    (close[i] < l4_aligned[i] or trend_12h_aligned[i] < 0))
        exit_short = (position == -1 and 
                     (close[i] > h4_aligned[i] or trend_12h_aligned[i] > 0))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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