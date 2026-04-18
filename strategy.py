#!/usr/bin/env python3
"""
6h_Momentum_Divergence_With_1D_RSI_Filter
Hypothesis: Use 6-hour price momentum (ROC) combined with daily RSI divergence to identify reversal points. Go long when 6H momentum turns positive while daily RSI shows bullish divergence, short when 6H momentum turns negative while daily RSI shows bearish divergence. Designed to work in both bull and bear markets by capturing momentum exhaustion and reversals. Targets 15-25 trades/year with position size 0.25.
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
    
    # Get 1D data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 6H ROC (Rate of Change) - 6 periods (36 hours)
    roc_6h = np.full(n, np.nan)
    for i in range(6, n):
        if close[i-6] != 0:
            roc_6h[i] = (close[i] - close[i-6]) / close[i-6] * 100
    
    # Calculate 6H momentum change (ROC crossing zero)
    roc_cross_up = np.zeros(n, dtype=bool)
    roc_cross_down = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(roc_6h[i]) and not np.isnan(roc_6h[i-1]):
            roc_cross_up[i] = roc_6h[i-1] <= 0 and roc_6h[i] > 0
            roc_cross_down[i] = roc_6h[i-1] >= 0 and roc_6h[i] < 0
    
    # Calculate daily RSI (14-period)
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(prices, np.nan)
        avg_loss = np.full_like(prices, np.nan)
        
        # First average
        if len(gain) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
        
        # Wilder's smoothing
        for i in range(period, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, 14)
    
    # Align daily RSI to 6h timeframe (wait for daily bar close)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate RSI divergence signals
    # Bullish divergence: price makes lower low, RSI makes higher low
    # Bearish divergence: price makes higher high, RSI makes lower high
    
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    # Look for divergences over 3-day window
    for i in range(2, n):
        if not np.isnan(rsi_1d_aligned[i]) and not np.isnan(rsi_1d_aligned[i-2]):
            # Check for price lows and RSI lows (bullish divergence)
            if (low[i] <= low[i-1] and low[i-1] >= low[i-2]) and \
               (rsi_1d_aligned[i] >= rsi_1d_aligned[i-1] and rsi_1d_aligned[i-1] <= rsi_1d_aligned[i-2]):
                if rsi_1d_aligned[i] < 40 and rsi_1d_aligned[i-2] < 40:  # Only in oversold territory
                    bullish_div[i] = True
            
            # Check for price highs and RSI highs (bearish divergence)
            if (high[i] >= high[i-1] and high[i-1] <= high[i-2]) and \
               (rsi_1d_aligned[i] <= rsi_1d_aligned[i-1] and rsi_1d_aligned[i-1] >= rsi_1d_aligned[i-2]):
                if rsi_1d_aligned[i] > 60 and rsi_1d_aligned[i-2] > 60:  # Only in overbought territory
                    bearish_div[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need sufficient data for ROC and divergence
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(roc_6h[i]) or np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: 6H momentum turns positive AND bullish RSI divergence
            if roc_cross_up[i] and bullish_div[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: 6H momentum turns negative AND bearish RSI divergence
            elif roc_cross_down[i] and bearish_div[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: momentum turns negative OR bearish divergence appears
            if roc_cross_down[i] or bearish_div[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: momentum turns positive OR bullish divergence appears
            if roc_cross_up[i] or bullish_div[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Momentum_Divergence_With_1D_RSI_Filter"
timeframe = "6h"
leverage = 1.0