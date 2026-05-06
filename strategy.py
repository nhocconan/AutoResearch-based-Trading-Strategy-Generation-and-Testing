#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Supertrend for trend direction and 1w RSI for momentum confirmation
# - Uses 1d Supertrend (ATR=10, multiplier=3) to determine trend direction
# - Uses 1w RSI(14) to confirm momentum: RSI > 50 for longs, RSI < 50 for shorts
# - Enters long when price crosses above Supertrend with bullish momentum (RSI > 50)
# - Enters short when price crosses below Supertrend with bearish momentum (RSI < 50)
# - Exits when price crosses back below/above Supertrend
# - Designed to capture trend continuation with momentum filter in both bull and bear markets
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "6h_1dSupertrend_1wRSI_Momentum"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Supertrend calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1d Supertrend (ATR=10, multiplier=3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(10)
    atr = np.zeros_like(close_1d)
    atr[9] = np.mean(tr[:10])  # Simple average for first value
    for i in range(10, len(tr)):
        atr[i] = (atr[i-1] * 9 + tr[i]) / 10  # Wilder's smoothing
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1d + low_1d) / 2 + 3 * atr
    basic_lb = (high_1d + low_1d) / 2 - 3 * atr
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(close_1d)
    final_lb = np.zeros_like(close_1d)
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    for i in range(1, len(close_1d)):
        if basic_ub[i] < final_ub[i-1] or close_1d[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close_1d[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend = np.zeros_like(close_1d)
    supertrend[0] = final_ub[0]
    for i in range(1, len(close_1d)):
        if supertrend[i-1] == final_ub[i-1]:
            if close_1d[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
            else:
                supertrend[i] = final_lb[i]
        else:
            if close_1d[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
            else:
                supertrend[i] = final_ub[i]
    
    # Calculate 1w RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Average gain and loss
    avg_gain = np.zeros_like(close_1w)
    avg_loss = np.zeros_like(close_1w)
    avg_gain[13] = np.mean(gain[1:14])  # First average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1w)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    # RS and RSI
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 1d Supertrend to 6h timeframe
    supertrend_6h = align_htf_to_ltf(prices, df_1d, supertrend)
    
    # Align 1w RSI to 6h timeframe
    rsi_6h = align_htf_to_ltf(prices, df_1w, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(supertrend_6h[i]) or np.isnan(rsi_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above Supertrend with bullish momentum (RSI > 50)
            if close[i] > supertrend_6h[i] and rsi_6h[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below Supertrend with bearish momentum (RSI < 50)
            elif close[i] < supertrend_6h[i] and rsi_6h[i] < 50:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Supertrend
            if close[i] < supertrend_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Supertrend
            if close[i] > supertrend_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals