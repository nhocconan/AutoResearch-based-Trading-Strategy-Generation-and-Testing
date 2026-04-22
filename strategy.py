#!/usr/bin/env python3
"""
Hypothesis: 1-day trading strategy using 1-week EMA trend filter and RSI momentum.
Long when price > weekly EMA50 and RSI(14) > 55.
Short when price < weekly EMA50 and RSI(14) < 45.
Exit when RSI returns to neutral zone (45-55).
Weekly EMA provides trend filter to avoid counter-trend trades.
RSI provides momentum signals with built-in mean reversion exit.
Designed for low trade frequency (~10-20/year) to avoid fee drag.
Works in bull markets (trend following) and bear markets (counter-trend bounces).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    rsi_input = prices['close'].values  # RSI uses close prices
    
    # Load 1-week data for EMA trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to daily timeframe (wait for weekly close)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily RSI(14)
    delta = pd.Series(rsi_input).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(rsi_values[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above weekly EMA50 and RSI > 55 (bullish momentum)
            if close[i] > ema_50_1w_aligned[i] and rsi_values[i] > 55:
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly EMA50 and RSI < 45 (bearish momentum)
            elif close[i] < ema_50_1w_aligned[i] and rsi_values[i] < 45:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when RSI returns to neutral zone (45-55)
            exit_signal = False
            
            if position == 1:
                # Exit long when RSI drops to 55 or below
                if rsi_values[i] <= 55:
                    exit_signal = True
            else:  # position == -1
                # Exit short when RSI rises to 45 or above
                if rsi_values[i] >= 45:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_WeeklyEMA50_RSI_Momentum"
timeframe = "1d"
leverage = 1.0