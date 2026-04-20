#!/usr/bin/env python3
# 4h_1d_RSI_Stochastic_Divergence
# Hypothesis: On 4h timeframe, trade reversals using 1d RSI and Stochastic divergences with volume confirmation.
# In both bull and bear markets, momentum divergences at extreme levels signal reversals.
# Uses 1d RSI(14) and Stochastic(14,3,3) to identify overbought/oversold conditions with bearish/bullish divergence.
# Volume confirmation filters false signals. Targets 20-40 trades/year by requiring confluence of divergence, extreme levels, and volume.

name = "4h_1d_RSI_Stochastic_Divergence"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(prices, np.nan)
    avg_loss = np.full_like(prices, np.nan)
    
    if len(prices) < period:
        return avg_gain
    
    # First average
    avg_gain[period-1] = np.mean(gain[:period])
    avg_loss[period-1] = np.mean(loss[:period])
    
    # Wilder smoothing
    for i in range(period, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = np.full_like(low, np.nan)
    highest_high = np.full_like(high, np.nan)
    
    for i in range(len(close)):
        if i < k_period - 1:
            continue
        lowest_low[i] = np.min(low[i-k_period+1:i+1])
        highest_high[i] = np.max(high[i-k_period+1:i+1])
    
    k_percent = np.divide((close - lowest_low), (highest_high - lowest_low), 
                          out=np.full_like(close, np.nan), where=(highest_high - lowest_low)!=0) * 100
    
    # D-period smoothing of K
    d_percent = np.full_like(k_percent, np.nan)
    for i in range(len(k_percent)):
        if i < d_period - 1 or np.isnan(k_percent[i-d_period+1:i+1]).any():
            continue
        d_percent[i] = np.nanmean(k_percent[i-d_period+1:i+1])
    
    return k_percent, d_percent

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14)
    rsi_1d = calculate_rsi(close_1d, 14)
    
    # Calculate 1d Stochastic(14,3,3)
    stoch_k_1d, stoch_d_1d = calculate_stochastic(high_1d, low_1d, close_1d, 14, 3)
    
    # Align 1d indicators to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    stoch_k_1d_aligned = align_htf_to_ltf(prices, df_1d, stoch_k_1d)
    stoch_d_1d_aligned = align_htf_to_ltf(prices, df_1d, stoch_d_1d)
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(stoch_k_1d_aligned[i]) or 
            np.isnan(stoch_d_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Bearish divergence: price makes higher high, RSI makes lower high (overbought)
            # Bullish divergence: price makes lower low, RSI makes higher low (oversold)
            
            # Check for bearish RSI divergence (sell signal)
            if (rsi_1d_aligned[i] > 70 and  # Overbought
                i >= 2 and
                close[i] > close[i-1] and close[i-1] > close[i-2] and  # Price making higher high
                rsi_1d_aligned[i] < rsi_1d_aligned[i-1] and rsi_1d_aligned[i-1] < rsi_1d_aligned[i-2]):  # RSI making lower high
                
                # Confirm with Stochastic overbought
                if stoch_k_1d_aligned[i] > 80 and stoch_d_1d_aligned[i] > 80:
                    if volume[i] > 1.5 * volume_ma[i]:  # Volume confirmation
                        signals[i] = -0.25
                        position = -1
            
            # Check for bullish RSI divergence (buy signal)
            elif (rsi_1d_aligned[i] < 30 and  # Oversold
                  i >= 2 and
                  close[i] < close[i-1] and close[i-1] < close[i-2] and  # Price making lower low
                  rsi_1d_aligned[i] > rsi_1d_aligned[i-1] and rsi_1d_aligned[i-1] > rsi_1d_aligned[i-2]):  # RSI making higher low
                
                # Confirm with Stochastic oversold
                if stoch_k_1d_aligned[i] < 20 and stoch_d_1d_aligned[i] < 20:
                    if volume[i] > 1.5 * volume_ma[i]:  # Volume confirmation
                        signals[i] = 0.25
                        position = 1
        
        elif position == 1:
            # Long exit: RSI returns to neutral or bearish divergence
            if (rsi_1d_aligned[i] >= 50 and rsi_1d_aligned[i-1] < 50) or \
               (rsi_1d_aligned[i] > 70 and 
                i >= 2 and
                close[i] > close[i-1] and close[i-1] > close[i-2] and
                rsi_1d_aligned[i] < rsi_1d_aligned[i-1] and rsi_1d_aligned[i-1] < rsi_1d_aligned[i-2]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral or bullish divergence
            if (rsi_1d_aligned[i] <= 50 and rsi_1d_aligned[i-1] > 50) or \
               (rsi_1d_aligned[i] < 30 and 
                i >= 2 and
                close[i] < close[i-1] and close[i-1] < close[i-2] and
                rsi_1d_aligned[i] > rsi_1d_aligned[i-1] and rsi_1d_aligned[i-1] > rsi_1d_aligned[i-2]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals