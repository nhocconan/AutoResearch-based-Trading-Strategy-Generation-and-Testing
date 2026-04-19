#!/usr/bin/env python3
"""
6h_RSI_Reversal_Filter
Hypothesis: RSI(14) reversals at weekly extremes with volume confirmation and trend filter
In bear markets, RSI < 20 on weekly timeframe indicates oversold conditions for bounce
In bull markets, RSI > 80 on weekly timeframe indicates overbought conditions for pullback
Volume confirmation on 6b ensures institutional participation
Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year)
"""

name = "6h_RSI_Reversal_Filter"
timeframe = "6h"
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
    
    # RSI(14) for 6h momentum
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        delta = np.insert(delta, 0, 0)  # same length as prices
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # First average
        if len(prices) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
            
            for i in range(period, len(prices)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Weekly RSI for extreme levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need enough for RSI calculation
        return np.zeros(n)
    
    weekly_rsi = calculate_rsi(df_1w['close'].values, 14)
    weekly_rsi_aligned = align_htf_to_ltf(prices, df_1w, weekly_rsi, additional_delay_bars=0)
    
    # 6h RSI for entry timing
    rsi_6h = calculate_rsi(close, 14)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_rsi_aligned[i]) or np.isnan(rsi_6h[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly RSI < 20 (oversold) and 6h RSI < 30 (pullback) with volume
            if (weekly_rsi_aligned[i] < 20 and 
                rsi_6h[i] < 30 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly RSI > 80 (overbought) and 6h RSI > 70 (pullback) with volume
            elif (weekly_rsi_aligned[i] > 80 and 
                  rsi_6h[i] > 70 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if weekly RSI > 50 (mean reversion) or 6h RSI > 70
            if (weekly_rsi_aligned[i] > 50) or (rsi_6h[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if weekly RSI < 50 (mean reversion) or 6h RSI < 30
            if (weekly_rsi_aligned[i] < 50) or (rsi_6h[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals