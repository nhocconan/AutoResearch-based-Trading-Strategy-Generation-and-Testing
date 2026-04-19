#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Filter
Hypothesis: 1-day KAMA identifies trend direction, RSI filters for momentum exhaustion, volume confirms institutional participation. 
KAMA adapts to market noise, reducing whipsaws in sideways markets. RSI > 50 for longs and < 50 for shorts ensures momentum alignment.
Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year). Works in bull/bear via trend-following logic.
"""

name = "1d_KAMA_Trend_With_RSI_Filter"
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
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio and Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # Placeholder for correct calc
    
    # Correct ER calculation: change over period / sum of absolute changes over period
    er_period = 10
    change_abs = np.abs(np.diff(close, prepend=close[0]))
    # For each point, calculate ER over er_period window
    er = np.zeros(n)
    er[:] = np.nan
    for i in range(er_period, n):
        price_change = np.abs(close[i] - close[i-er_period])
        sum_abs_changes = np.sum(change_abs[i-er_period+1:i+1])
        if sum_abs_changes > 0:
            er[i] = price_change / sum_abs_changes
        else:
            er[i] = 0
    
    # Smoothing constant
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1))**2
    # KAMA calculation
    kama = np.zeros(n)
    kama[:] = np.nan
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14) for momentum filter
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    # Align KAMA to daily timeframe (already daily, but ensuring alignment)
    kama_aligned = kama  # Already on 1d timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI > 50 (bullish momentum), volume confirmation
            if (close[i] > kama_aligned[i] and 
                rsi[i] > 50 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50 (bearish momentum), volume confirmation
            elif (close[i] < kama_aligned[i] and 
                  rsi[i] < 50 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price below KAMA or RSI < 40 (losing momentum)
            if (close[i] < kama_aligned[i]) or (rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price above KAMA or RSI > 60 (losing momentum)
            if (close[i] > kama_aligned[i]) or (rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals