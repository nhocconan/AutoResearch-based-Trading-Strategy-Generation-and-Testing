#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_R1S1_Breakout_With_Pullback_Entry
Hypothesis: Breakout with pullback confirmation reduces false signals and improves edge.
Uses 4h for trend (EMA21>EMA50), 1d for support/resistance (Camarilla), and 1h for entry.
Enters on pullback to breakout level after initial break, not on the break itself.
Long: Price breaks above 1d R1, then pulls back to R1 with volume and 4h uptrend.
Short: Price breaks below 1d S1, then pulls back to S1 with volume and 4h downtrend.
Exit when price crosses 1d pivot point. Target: 15-30 trades/year per symbol.
Works in bull/bear by following higher timeframe trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1, and pivot point (PP)
    rang = prev_high - prev_low
    r1 = prev_close + 1.1 * rang / 12
    s1 = prev_close - 1.1 * rang / 12
    pp = (prev_high + prev_low + prev_close) / 3
    
    # Align to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate EMA21 and EMA50 on 4h
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align to 1h timeframe
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    breakout_high = np.full(n, np.nan)  # Track breakout levels
    breakout_low = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema21_4h_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # 4h trend filter: EMA21 > EMA50 for uptrend, EMA21 < EMA50 for downtrend
        uptrend = ema21_4h_aligned[i] > ema50_4h_aligned[i]
        downtrend = ema21_4h_aligned[i] < ema50_4h_aligned[i]
        
        # Track breakout levels
        if i > 0:
            breakout_high[i] = breakout_high[i-1]
            breakout_low[i] = breakout_low[i-1]
        
        # Update breakout levels on new breaks
        if position == 0:
            if price > r1_aligned[i] and not np.isnan(r1_aligned[i]):
                breakout_high[i] = r1_aligned[i]  # Mark breakout level
            if price < s1_aligned[i] and not np.isnan(s1_aligned[i]):
                breakout_low[i] = s1_aligned[i]   # Mark breakout level
        
        if position == 0:
            # Long: pullback to breakout level after upward break
            if (not np.isnan(breakout_high[i]) and 
                abs(price - breakout_high[i]) < 0.001 * breakout_high[i] and  # Near breakout level
                volume_ok and uptrend):
                signals[i] = 0.20
                position = 1
            # Short: pullback to breakout level after downward break
            elif (not np.isnan(breakout_low[i]) and 
                  abs(price - breakout_low[i]) < 0.001 * breakout_low[i] and  # Near breakout level
                  volume_ok and downtrend):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below pivot point
            if price < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses back above pivot point
            if price > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_Camarilla_R1S1_Breakout_With_Pullback_Entry"
timeframe = "1h"
leverage = 1.0