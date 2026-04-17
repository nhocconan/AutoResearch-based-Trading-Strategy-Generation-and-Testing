#!/usr/bin/env python3
"""
6h_RSI_Overbought_Oversold_With_Stochastic_Confirmation
Hypothesis: 6h RSI extremes (overbought >70, oversold <30) confirmed by 6h Stochastic Oscillator to reduce false signals in trending markets. Works in both bull (mean reversion in uptrend) and bear (mean reversion in downtrend) regimes. Uses 60-period RSI to reduce noise and 14-period Stochastic with smoothing. Entry when RSI crosses extreme with Stochastic confirmation, exit when RSI returns to neutral zone (40-60). Position size 0.25 to manage drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # RSI(60) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/period)
    alpha = 1.0 / 60
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, n):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator (14,3,3)
    lowest_low = np.zeros(n)
    highest_high = np.zeros(n)
    
    for i in range(n):
        start_idx = max(0, i - 13)
        lowest_low[i] = np.min(low[start_idx:i+1])
        highest_high[i] = np.max(high[start_idx:i+1])
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    stoch_k = np.where(denominator != 0, 100 * (close - lowest_low) / denominator, 0)
    
    # Smooth %K to get %D (3-period SMA of %K)
    stoch_d = np.zeros(n)
    for i in range(n):
        if i < 2:
            stoch_d[i] = stoch_k[i] if i == 0 else (stoch_k[0] + stoch_k[1]) / 2
        else:
            stoch_d[i] = (stoch_k[i-2] + stoch_k[i-1] + stoch_k[i]) / 3
    
    signals = np.zeros(n)
    
    # Warmup: covers RSI and Stochastic calculations
    warmup = 80
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Entry conditions
        if position == 0:
            # Long: RSI crosses above 30 from below AND Stochastic %K crosses above %D
            if rsi[i-1] <= 30 and rsi[i] > 30 and stoch_k[i-1] <= stoch_d[i-1] and stoch_k[i] > stoch_d[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: RSI crosses below 70 from above AND Stochastic %K crosses below %D
            elif rsi[i-1] >= 70 and rsi[i] < 70 and stoch_k[i-1] >= stoch_d[i-1] and stoch_k[i] < stoch_d[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions
        elif position == 1:
            # Exit long when RSI returns to neutral zone (above 40)
            if rsi[i] >= 40:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when RSI returns to neutral zone (below 60)
            if rsi[i] <= 60:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI_Overbought_Oversold_With_Stochastic_Confirmation"
timeframe = "6h"
leverage = 1.0