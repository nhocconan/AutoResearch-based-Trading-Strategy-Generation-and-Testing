#!/usr/bin/env python3
# 4h_RSI_2_45_Stochastic_Bullish_Cross_With_Volume
# Hypothesis: 4-hour RSI(2) crossing above 45 (reversal from oversold) combined with
# bullish Stochastic crossover (%K > %D) and volume confirmation. RSI(2) captures
# short-term momentum reversals effectively. Volume ensures institutional participation.
# Works in bull markets via momentum continuations and in bear markets via oversold
# bounces. Target: 20-40 trades/year to avoid fee drag.

name = "4h_RSI_2_45_Stochastic_Bullish_Cross_With_Volume"
timeframe = "4h"
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
    
    # RSI(2) - fast RSI for early reversal signals
    def calculate_rsi(close_prices, period=2):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        # First average
        if len(close_prices) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
            
            for i in range(period, len(close_prices)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Stochastic Oscillator
    def calculate_stochastic(high_prices, low_prices, close_prices, k_period=14, d_period=3):
        # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
        lowest_low = np.zeros_like(low_prices)
        highest_high = np.zeros_like(high_prices)
        
        for i in range(len(close_prices)):
            start_idx = max(0, i - k_period + 1)
            lowest_low[i] = np.min(low_prices[start_idx:i+1])
            highest_high[i] = np.max(high_prices[start_idx:i+1])
        
        # Avoid division by zero
        denominator = highest_high - lowest_low
        k_percent = np.where(denominator != 0, 
                            (close_prices - lowest_low) / denominator * 100, 0)
        
        # %D = SMA of %K
        d_percent = np.zeros_like(k_percent)
        for i in range(len(k_percent)):
            start_idx = max(0, i - d_period + 1)
            if i >= d_period - 1:
                d_percent[i] = np.mean(k_percent[start_idx:i+1])
            else:
                d_percent[i] = k_percent[i]  # Not enough data yet
        
        return k_percent, d_percent
    
    # Calculate indicators
    rsi_2 = calculate_rsi(close, 2)
    stoch_k, stoch_d = calculate_stochastic(high, low, close, 14, 3)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    start_idx = max(30, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_2[i]) or np.isnan(stoch_k[i]) or np.isnan(stoch_d[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: RSI(2) > 45 AND bullish Stochastic cross AND volume
        rsi_condition = rsi_2[i] > 45
        stoch_cross = stoch_k[i] > stoch_d[i] and stoch_k[i-1] <= stoch_d[i-1]
        
        if position == 0:
            if rsi_condition and stoch_cross and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
        elif position == 1:
            # Exit: RSI(2) < 55 (overbought) OR bearish Stochastic cross
            rsi_exit = rsi_2[i] < 55
            stoch_cross_down = stoch_k[i] < stoch_d[i] and stoch_k[i-1] >= stoch_d[i-1]
            
            if rsi_exit or stoch_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals