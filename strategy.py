#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Stochastic Oscillator + 1d RSI Trend Filter
# - Stochastic Oscillator (14,3,3) on 6h for momentum reversal signals
# - Long when %K < 20 (oversold) and 1d RSI > 50 (uptrend bias)
# - Short when %K > 80 (overbought) and 1d RSI < 50 (downtrend bias)
# - Stochastic captures short-term reversals; daily RSI filters for intermediate trend
# - Designed for 6h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on 1d timeframe
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_14_1d.values
    
    # Align 1d RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate Stochastic Oscillator (14,3,3) on 6h timeframe
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * (close_6h - lowest_low_14) / (highest_high_14 - lowest_low_14)
    # Smooth %K to get %D (3-period SMA of %K)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after Stochastic %D warmup
        # Skip if NaN in indicators
        if np.isnan(d_percent[i]) or np.isnan(rsi_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        k = d_percent[i]  # Using smoothed %D for signals
        rsi = rsi_1d_aligned[i]
        
        if position == 0:
            # Long entry: Stochastic oversold (< 20) + 1d RSI > 50 (uptrend)
            if k < 20 and rsi > 50:
                signals[i] = 0.25
                position = 1
            # Short entry: Stochastic overbought (> 80) + 1d RSI < 50 (downtrend)
            elif k > 80 and rsi < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Stochastic rises above 50 or RSI turns bearish
            if k > 50 or rsi < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Stochastic falls below 50 or RSI turns bullish
            if k < 50 or rsi > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Stochastic_1dRSI_TrendFilter"
timeframe = "6h"
leverage = 1.0