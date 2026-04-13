#!/usr/bin/env python3
"""
4h_1d_1w_Camarilla_Pivot_Breakout_With_Trend_Filter
Hypothesis: Combines Camarilla pivot levels on 1d with 1w trend filter and volume confirmation.
In trending markets (price above/below 1w EMA200), takes long/short at Camarilla H4/L4 levels with volume > 1.5x average.
In ranging markets, avoids trades to reduce false breakouts.
Works in both bull and bear markets by trading breakouts in the direction of higher timeframe trend.
Target: 20-50 trades/year on 4h (80-200 total over 4 years).
"""

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
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # H4 = close + 1.1 * (high - low) / 2
    # L4 = close - 1.1 * (high - low) / 2
    H4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    L4 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA200 on weekly
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all signals to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation on 4h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(H4_aligned[i]) or 
            np.isnan(L4_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from weekly EMA200
        uptrend = close[i] > ema_200_1w_aligned[i]
        downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Entry conditions: price at Camarilla level with volume expansion and trend alignment
        if volume_expansion[i]:
            # Long when price crosses above H4 in uptrend
            if uptrend and close[i] > H4_aligned[i] and (i == 0 or close[i-1] <= H4_aligned[i]):
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            # Short when price crosses below L4 in downtrend
            elif downtrend and close[i] < L4_aligned[i] and (i == 0 or close[i-1] >= L4_aligned[i]):
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            # Hold position
            elif position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        else:
            # No volume expansion - exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_1w_Camarilla_Pivot_Breakout_With_Trend_Filter"
timeframe = "4h"
leverage = 1.0