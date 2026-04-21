#!/usr/bin/env python3
"""
6h_RSI_Trend_Filter_V1
Hypothesis: Use 1d RSI for trend bias (bullish if RSI>50, bearish if RSI<50).
On 6h, enter long when RSI(14) crosses above 30 with volume confirmation.
Enter short when RSI(14) crosses below 70 with volume confirmation.
Exit on opposite RSI cross or trend reversal.
Designed for 6h timeframe with 1d filter to limit trades to ~15-30/year.
Works in bull markets by buying dips and in bear markets by selling rallies.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    if len(close) >= period:
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        
        for i in range(period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    rsi_1d = calculate_rsi(close_1d, 14)
    # Bullish if RSI > 50, bearish if RSI < 50
    bias = np.where(rsi_1d > 50, 1, np.where(rsi_1d < 50, -1, 0))
    bias_aligned = align_htf_to_ltf(prices, df_1d, bias)
    
    # Calculate 6h RSI
    close = prices['close'].values
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if np.isnan(rsi[i]) or np.isnan(bias_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = prices['volume'].iloc[i] > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: bullish bias + RSI crosses above 30 + volume
            if (bias_aligned[i] > 0 and 
                rsi[i] > 30 and rsi[i-1] <= 30 and 
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish bias + RSI crosses below 70 + volume
            elif (bias_aligned[i] < 0 and 
                  rsi[i] < 70 and rsi[i-1] >= 70 and 
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish bias or RSI crosses below 70
            if bias_aligned[i] < 0 or (rsi[i] < 70 and rsi[i-1] >= 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish bias or RSI crosses above 30
            if bias_aligned[i] > 0 or (rsi[i] > 30 and rsi[i-1] <= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI_Trend_Filter_V1"
timeframe = "6h"
leverage = 1.0