#!/usr/bin/env python3
# 4h_RSI_Stochastic_Confluence
# Hypothesis: Combining RSI mean reversion with Stochastic oscillator on 4h timeframe provides high-probability entries during pullbacks in trending markets. RSI identifies overbought/oversold conditions while Stochastic confirms momentum exhaustion. Volume filter ensures institutional participation. Designed to work in both bull and bear markets by capturing mean reversion within the dominant trend, reducing whipsaw and improving risk-reward.

name = "4h_RSI_Stochastic_Confluence"
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
    
    # RSI calculation
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        delta = np.concatenate([[np.nan], delta])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # Wilder's smoothing
        avg_gain[period] = np.nanmean(gain[1:period+1])
        avg_loss[period] = np.nanmean(loss[1:period+1])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.divide(avg_gain, avg_loss, out=np.full_like(close, np.nan), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Stochastic oscillator calculation
    def calculate_stochastic(high, low, close, k_period=14, d_period=3):
        lowest_low = np.zeros_like(low)
        highest_high = np.zeros_like(high)
        
        for i in range(len(close)):
            if i < k_period:
                lowest_low[i] = np.nan
                highest_high[i] = np.nan
            else:
                lowest_low[i] = np.min(low[i-k_period+1:i+1])
                highest_high[i] = np.max(high[i-k_period+1:i+1])
        
        k_percent = np.divide((close - lowest_low), (highest_high - lowest_low), 
                              out=np.full_like(close, np.nan), where=(highest_high-lowest_low)!=0) * 100
        
        # Smoothed K (D)
        d_percent = np.full_like(close, np.nan)
        for i in range(len(close)):
            if i < k_period + d_period - 1:
                d_percent[i] = np.nan
            else:
                d_percent[i] = np.nanmean(k_percent[i-d_period+1:i+1])
        
        return k_percent, d_percent
    
    # Get 4h data for indicators
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate RSI on 4h close
    rsi = calculate_rsi(df_4h['close'].values, 14)
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi)
    
    # Calculate Stochastic on 4h OHLC
    stoch_k, stoch_d = calculate_stochastic(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        14, 3
    )
    stoch_k_aligned = align_htf_to_ltf(prices, df_4h, stoch_k)
    stoch_d_aligned = align_htf_to_ltf(prices, df_4h, stoch_d)
    
    # Volume confirmation: volume > 1.2 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.2)
    
    # ATR for stop loss
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or np.isnan(stoch_k_aligned[i]) or 
            np.isnan(stoch_d_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) AND Stochastic K crosses above D (bullish momentum)
            if (rsi_aligned[i] < 30 and 
                stoch_k_aligned[i] > stoch_d_aligned[i] and 
                stoch_k_aligned[i-1] <= stoch_d_aligned[i-1] and  # crossover
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) AND Stochastic K crosses below D (bearish momentum)
            elif (rsi_aligned[i] > 70 and 
                  stoch_k_aligned[i] < stoch_d_aligned[i] and 
                  stoch_k_aligned[i-1] >= stoch_d_aligned[i-1] and  # crossover
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if RSI overbought OR Stochastic bearish crossover OR ATR stop
            if (rsi_aligned[i] > 70) or \
               (stoch_k_aligned[i] < stoch_d_aligned[i] and stoch_k_aligned[i-1] >= stoch_d_aligned[i-1]) or \
               (close[i] < close[i-1] - 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if RSI oversold OR Stochastic bullish crossover OR ATR stop
            if (rsi_aligned[i] < 30) or \
               (stoch_k_aligned[i] > stoch_d_aligned[i] and stoch_k_aligned[i-1] <= stoch_d_aligned[i-1]) or \
               (close[i] > close[i-1] + 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals