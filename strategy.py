#!/usr/bin/env python3
"""
Hypothesis: 4-hour Bollinger Squeeze with 1-day RSI momentum.
Long when price breaks above upper Bollinger band during low volatility (squeeze) and 1-day RSI > 50.
Short when price breaks below lower Bollinger band during low volatility and 1-day RSI < 50.
Exit when price returns to middle Bollinger band (20-period SMA).
Bollinger Squeeze identifies low volatility breakouts; 1-day RSI filters momentum direction.
Designed for low trade frequency by requiring volatility contraction before breakout.
Works in both bull and bear markets by capturing volatility expansion moves in direction of higher timeframe momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1-day data for RSI filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 14-period RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Bollinger Bands (20, 2)
    ma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    
    # Bollinger Band Width for squeeze detection (normalized by middle band)
    bb_width = (upper - lower) / ma20
    # Squeeze condition: BB width below its 50-period minimum (volatility contraction)
    bb_width_min = pd.Series(bb_width).rolling(window=50, min_periods=50).min().values
    squeeze = bb_width < bb_width_min
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ma20[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(squeeze[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper band during squeeze and 1-day RSI > 50
            if (close[i] > upper[i] and squeeze[i] and rsi_1d_aligned[i] > 50):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band during squeeze and 1-day RSI < 50
            elif (close[i] < lower[i] and squeeze[i] and rsi_1d_aligned[i] < 50):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to middle Bollinger band (20-period SMA)
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below middle band
                if close[i] < ma20[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price rises above middle band
                if close[i] > ma20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Bollinger_Squeeze_1dRSI_Momentum"
timeframe = "4h"
leverage = 1.0