#!/usr/bin/env python3
# 6h_sma_crossover_momentum_v1
# Hypothesis: SMA crossover with momentum confirmation on 6h timeframe.
# Long when 20-period SMA crosses above 50-period SMA with RSI > 50.
# Short when 20-period SMA crosses below 50-period SMA with RSI < 50.
# Exit when opposite crossover occurs.
# Uses trend confirmation from 1d timeframe: only trade in direction of 1d 50-period SMA.
# Designed to work in both bull and bear markets by following higher timeframe trend.
# Target: 20-40 trades/year with strict entry conditions to avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_sma_crossover_momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate SMAs on 6h data
    sma_fast_period = 20
    sma_slow_period = 50
    
    close_series = pd.Series(close)
    sma_fast = close_series.rolling(window=sma_fast_period, min_periods=sma_fast_period).mean().values
    sma_slow = close_series.rolling(window=sma_slow_period, min_periods=sma_slow_period).mean().values
    
    # Calculate RSI for momentum confirmation
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = pd.Series(df_1d['close'].values)
    sma_1d_50 = close_1d.rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(sma_fast_period, sma_slow_period, rsi_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma_fast[i]) or np.isnan(sma_slow[i]) or 
            np.isnan(rsi[i]) or np.isnan(sma_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Fast SMA crosses below slow SMA
            if sma_fast[i] < sma_slow[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Fast SMA crosses above slow SMA
            if sma_fast[i] > sma_slow[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Fast SMA crosses above slow SMA with RSI > 50 and above 1d SMA
            if (sma_fast[i] > sma_slow[i] and 
                sma_fast[i-1] <= sma_slow[i-1] and  # crossover just happened
                rsi[i] > 50 and 
                close[i] > sma_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Fast SMA crosses below slow SMA with RSI < 50 and below 1d SMA
            elif (sma_fast[i] < sma_slow[i] and 
                  sma_fast[i-1] >= sma_slow[i-1] and  # crossover just happened
                  rsi[i] < 50 and 
                  close[i] < sma_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals