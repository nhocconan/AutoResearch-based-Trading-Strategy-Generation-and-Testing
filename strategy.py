#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h_4h_1d_rsi_momentum_v1
# Uses 4h RSI(14) for trend direction and 1d RSI(14) for regime filter.
# Long when 4h RSI crosses above 50 and 1d RSI > 40 (bullish regime).
# Short when 4h RSI crosses below 50 and 1d RSI < 60 (bearish regime).
# Exits when 4h RSI returns to 50.
# Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
# Works in trending markets via RSI momentum and in ranging markets via mean reversion to 50.

name = "1h_4h_1d_rsi_momentum_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for RSI calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h RSI(14)
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (equivalent to EMA with alpha=1/14)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d RSI(14)
    close_1d = df_1d['close'].values
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/14, adjust=False).mean().values
    rs_1d = avg_gain_1d / (avg_loss_1d + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    
    # Align HTF indicators to 1h timeframe
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(rsi_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        rsi4 = rsi_4h_aligned[i]
        rsi1d = rsi_1d_aligned[i]
        
        # Long signal: 4h RSI crosses above 50 and 1d RSI > 40 (bullish regime)
        if rsi4 > 50 and rsi1d > 40 and position != 1:
            position = 1
            signals[i] = 0.20
        # Short signal: 4h RSI crosses below 50 and 1d RSI < 60 (bearish regime)
        elif rsi4 < 50 and rsi1d < 60 and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit conditions: 4h RSI returns to 50 (mean reversion)
        elif position == 1 and rsi4 <= 50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and rsi4 >= 50:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals