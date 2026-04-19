#!/usr/bin/env python3
"""
6h_Camarilla_Pivot_Confluence_V1
Hypothesis: 6s Camarilla R1/S1 breakout with 1d RSI filter and volume confirmation
- Combines intraday support/resistance (Camarilla) with daily momentum (RSI)
- Requires volume > 1.5x 20-period average to avoid false breakouts
- Works in bull/bear via RSI filter (RSI>50 for longs, RSI<50 for shorts)
- Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
"""

name = "6h_Camarilla_Pivot_Confluence_V1"
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
    
    # 6h data for RSI calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # RSI(14) on 6h data
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        if len(close_prices) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
            
            for i in range(period, len(close_prices)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_6h = calculate_rsi(df_6h['close'].values, 14)
    rsi_6h_aligned = align_htf_to_ltf(prices, df_6h, rsi_6h)
    
    # Previous day's Camarilla levels (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    ph = df_1d['high'].shift(1).values  # Previous day high
    pl = df_1d['low'].shift(1).values   # Previous day low
    pc = df_1d['close'].shift(1).values # Previous day close
    
    # Camarilla calculations
    rang = ph - pl
    r1 = pc + (rang * 1.1 / 12)
    s1 = pc - (rang * 1.1 / 12)
    r4 = pc + (rang * 1.1 / 2)
    s4 = pc - (rang * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_6h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # RSI filter: RSI > 50 for long bias, RSI < 50 for short bias
        rsi_bullish = rsi_6h_aligned[i] > 50
        rsi_bearish = rsi_6h_aligned[i] < 50
        
        if position == 0:
            # Long: price breaks above R1 with volume and RSI bullish
            if (close[i] > r1_aligned[i] and 
                volume_confirm[i] and 
                rsi_bullish):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and RSI bearish
            elif (close[i] < s1_aligned[i] and 
                  volume_confirm[i] and 
                  rsi_bearish):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 or RSI turns bearish
            if (close[i] < s1_aligned[i]) or (rsi_6h_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 or RSI turns bullish
            if (close[i] > r1_aligned[i]) or (rsi_6h_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals