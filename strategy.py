#!/usr/bin/env python3
"""
4h_RSI_Pullback_With_Volume_and_Trend_Filter
Hypothesis: RSI pullback strategy on 4h timeframe with volume confirmation and trend filter.
- RSI(14) < 30 for long entries, > 70 for short entries (oversold/overbought)
- Price must be above/below 50-period EMA for trend alignment
- Volume must be above 20-period average for confirmation
- Designed for 4h timeframe targeting 20-50 trades/year (80-200 total over 4 years)
- Works in bull/bear via EMA trend filter - only trade with the trend
"""

name = "4h_RSI_Pullback_With_Volume_and_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) calculation
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # EMA(50) for trend filter
    def calculate_ema(prices, period):
        ema = np.zeros_like(prices)
        multiplier = 2 / (period + 1)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema
    
    # Calculate indicators
    rsi = calculate_rsi(close, 14)
    ema_50 = calculate_ema(close, 50)
    
    # Volume confirmation: volume > 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi[i]) or np.isnan(ema_50[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold (< 30) + price above EMA50 (uptrend) + volume confirmation
            if (rsi[i] < 30 and 
                close[i] > ema_50[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (> 70) + price below EMA50 (downtrend) + volume confirmation
            elif (rsi[i] > 70 and 
                  close[i] < ema_50[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if RSI overbought (> 70) or price breaks below EMA50
            if (rsi[i] > 70) or (close[i] < ema_50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if RSI oversold (< 30) or price breaks above EMA50
            if (rsi[i] < 30) or (close[i] > ema_50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals