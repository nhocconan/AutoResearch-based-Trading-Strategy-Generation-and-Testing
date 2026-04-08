#!/usr/bin/env python3
"""
1h_4h_1d_rsi_momentum_v1
Hypothesis: Use 4h RSI for momentum direction and 1d RSI for regime filter, with 1h for precise entry timing.
Enter long when 4h RSI > 50 (bullish momentum) and 1d RSI > 40 (not oversold) and 1h RSI crosses above 50.
Enter short when 4h RSI < 50 (bearish momentum) and 1d RSI < 60 (not overbought) and 1h RSI crosses below 50.
Exit when momentum reverses or RSI reaches extreme levels.
Designed for ~20-30 trades/year to avoid fee drag, works in bull/bear via momentum filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_rsi_momentum_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for momentum
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h RSI(14) for momentum direction
    close_4h = df_4h['close'].values
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_4h = avg_gain_4h / (avg_loss_4h + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d RSI(14) for regime filter
    close_1d = df_1d['close'].values
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1d = avg_gain_1d / (avg_loss_1d + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1h RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: momentum turns bearish or RSI overbought
            if rsi_4h_aligned[i] < 50 or rsi[i] >= 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: momentum turns bullish or RSI oversold
            if rsi_4h_aligned[i] > 50 or rsi[i] <= 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: 4h bullish momentum, 1d not oversold, 1h RSI crosses above 50
            if (rsi_4h_aligned[i] > 50 and 
                rsi_1d_aligned[i] > 40 and 
                rsi[i] > 50 and rsi[max(0, i-1)] <= 50):
                position = 1
                signals[i] = 0.20
            # Short entry: 4h bearish momentum, 1d not overbought, 1h RSI crosses below 50
            elif (rsi_4h_aligned[i] < 50 and 
                  rsi_1d_aligned[i] < 60 and 
                  rsi[i] < 50 and rsi[max(0, i-1)] >= 50):
                position = -1
                signals[i] = -0.20
    
    return signals