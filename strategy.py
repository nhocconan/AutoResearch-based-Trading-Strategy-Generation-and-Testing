#!/usr/bin/env python3
# 1d_weekly_rsi_momentum_v1
# Hypothesis: Weekly RSI extremes with daily momentum filter captures major trend reversals in both bull and bear markets.
# Weekly RSI < 30 indicates oversold conditions on weekly timeframe, suggesting potential bounce.
# Weekly RSI > 70 indicates overbought conditions, suggesting potential pullback.
# Daily price momentum (close > open) confirms short-term bias in direction of weekly signal.
# Designed for low frequency (10-25 trades/year) to minimize fee drag while capturing significant moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_rsi_momentum_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    
    # Get weekly data for RSI calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI (14-period)
    weekly_close = df_weekly['close'].values
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to daily timeframe (wait for weekly close)
    rsi_aligned = align_htf_to_ltf(prices, df_weekly, rsi)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if RSI data is not available
        if np.isnan(rsi_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: weekly RSI returns to neutral (> 50) or daily momentum turns negative
            if rsi_aligned[i] > 50 or close[i] < open_price[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: weekly RSI returns to neutral (< 50) or daily momentum turns positive
            if rsi_aligned[i] < 50 or close[i] > open_price[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: weekly RSI oversold (< 30) with positive daily momentum
            if rsi_aligned[i] < 30 and close[i] > open_price[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: weekly RSI overbought (> 70) with negative daily momentum
            elif rsi_aligned[i] > 70 and close[i] < open_price[i]:
                position = -1
                signals[i] = -0.25
    
    return signals