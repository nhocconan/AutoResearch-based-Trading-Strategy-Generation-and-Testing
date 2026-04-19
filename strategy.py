#!/usr/bin/env python3
"""
1d_WeeklyPivot_MeanReversion
Hypothesis: Daily mean reversion at weekly pivot points with volume confirmation and RSI filter.
Weekly pivots act as strong support/resistance in BTC/ETH due to institutional order flow.
RSI < 30 for long, > 70 for short ensures oversold/overbought conditions.
Volume confirmation filters for institutional participation.
Works in both bull/bear via mean reversion at key weekly levels.
Target: 15-25 trades/year to minimize fee drag.
"""

name = "1d_WeeklyPivot_MeanReversion"
timeframe = "1d"
leverage = 1.0

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
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for pivot calculation
    ph = df_1w['high'].shift(1).values   # Previous week high
    pl = df_1w['low'].shift(1).values    # Previous week low
    pc = df_1w['close'].shift(1).values  # Previous week close
    
    # Standard pivot point calculation
    pp = (ph + pl + pc) / 3.0           # Pivot point
    r1 = 2 * pp - pl                    # Resistance 1
    s1 = 2 * pp - ph                    # Support 1
    r2 = pp + (ph - pl)                 # Resistance 2
    s2 = pp - (ph - pl)                 # Support 2
    
    # Align weekly pivots to daily timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # RSI(14) for overbought/oversold conditions
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.full_like(prices, np.nan)
        avg_loss = np.full_like(prices, np.nan)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Volume confirmation: volume > 1.3 * 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price at or below S1 with RSI oversold and volume
            if (close[i] <= s1_aligned[i] and 
                rsi[i] < 30 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price at or above R1 with RSI overbought and volume
            elif (close[i] >= r1_aligned[i] and 
                  rsi[i] > 70 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit at pivot point or RSI overbought
            if (close[i] >= pp_aligned[i]) or (rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit at pivot point or RSI oversold
            if (close[i] <= pp_aligned[i]) or (rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals