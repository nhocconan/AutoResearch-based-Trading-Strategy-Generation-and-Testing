#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_Momentum
Hypothesis: Combines KAMA trend direction with RSI momentum and volume confirmation. 
KAMA adapts to market conditions, providing reliable trend signals in both trending and ranging markets. 
RSI filters for momentum strength, while volume confirms institutional participation. 
Designed for low trade frequency (<30/year) to minimize fee burn while capturing strong moves.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 4h data for KAMA calculation (using same timeframe as primary)
    # We'll calculate KAMA on 4h closes directly
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def calculate_kama(close_prices, length=10, fast=2, slow=30):
        # Calculate efficiency ratio
        change = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        volatility = np.abs(np.diff(close_prices))
        
        # Avoid division by zero
        er = np.zeros_like(close_prices)
        for i in range(1, len(close_prices)):
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        
        # Smooth ER
        er_smoothed = np.zeros_like(close_prices)
        er_smoothed[0] = er[0]
        for i in range(1, len(close_prices)):
            er_smoothed[i] = 0.1 * er[i] + 0.9 * er_smoothed[i-1]
        
        # Calculate smoothing constant
        sc = (er_smoothed * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # Calculate KAMA
        kama = np.zeros_like(close_prices)
        kama[0] = close_prices[0]
        for i in range(1, len(close_prices)):
            kama[i] = kama[i-1] + sc[i] * (close_prices[i] - kama[i-1])
        
        return kama
    
    # Calculate KAMA on 4h closes
    kama = calculate_kama(close, length=10, fast=2, slow=30)
    
    # Calculate RSI
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Calculate average gain and loss
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        # Calculate RS and RSI
        rs = np.zeros_like(close_prices)
        rsi = np.zeros_like(close_prices)
        for i in range(period, len(close_prices)):
            if avg_loss[i] != 0:
                rs[i] = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi[i] = 100
        
        return rsi
    
    rsi = calculate_rsi(close, period=14)
    
    # Volume confirmation: volume above 20-period average
    volume_ma = np.zeros_like(volume)
    for i in range(20, len(volume)):
        volume_ma[i] = np.mean(volume[i-20:i])
    volume_ma[:20] = volume_ma[20]  # Fill initial values
    volume_confirm = volume > volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to KAMA
        uptrend = close[i] > kama[i]
        downtrend = close[i] < kama[i]
        
        # Momentum filter: RSI in productive range (not overbought/oversold extremes)
        rsi_momentum = (rsi[i] > 40) & (rsi[i] < 80)  # Avoid extremes
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Entry conditions: KAMA direction with momentum and volume
        long_entry = uptrend & rsi_momentum & vol_confirm
        short_entry = downtrend & rsi_momentum & vol_confirm
        
        # Exit conditions: trend reversal or momentum exhaustion
        long_exit = (~uptrend) | (rsi[i] > 75)  # Exit on trend change or overbought
        short_exit = (~downtrend) | (rsi[i] < 25)  # Exit on trend change or oversold
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Direction_RSI_Momentum"
timeframe = "4h"
leverage = 1.0