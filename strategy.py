# 4h_1d_KAMA20_RSI14_Volume
# Hypothesis: 4h timeframe with KAMA20 trend direction, RSI14 for momentum, and volume confirmation. 
# KAMA adapts to market noise, reducing whipsaws in sideways markets. RSI14 filters momentum extremes.
# Volume confirms institutional participation. Works in bull/bear by following adaptive trend.
# Targets 20-50 trades/year (80-200 total over 4 years) with strict entry conditions.
# Uses 1d for higher timeframe context but focuses on 4h execution.
name = "4h_1d_KAMA20_RSI14_Volume"
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) 20-period
    # ER (Efficiency Ratio) = abs(close - close[9]) / sum(abs(close - close.shift(1))) over 10 periods
    change = np.abs(np.subtract(close[10:], close[:-10]))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) > 1 else np.zeros_like(close)
    # Simplified ER calculation for efficiency
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        price_change = np.abs(close[i] - close[i-10])
        price_volatility = np.sum(np.abs(np.diff(close[i-10:i+1])))
        er[i] = price_change / price_volatility if price_volatility != 0 else 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # 2-period EMA
    slow_sc = 2 / (30 + 1)  # 30-period EMA
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start with first close
    for i in range(10, len(close)):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[13] = np.mean(gain[1:14]) if len(gain) >= 14 else np.nan
    avg_loss[13] = np.mean(loss[1:14]) if len(loss) >= 14 else np.nan
    
    for i in range(14, len(close)):
        if not np.isnan(avg_gain[i-1]) and not np.isnan(avg_loss[i-1]):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(close, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.3 * 20-period average
    volume_ma = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        volume_ma[i] = np.mean(volume[i-19:i+1])
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA AND RSI > 50 (bullish momentum) with volume
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA AND RSI < 50 (bearish momentum) with volume
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA OR RSI < 40 (losing momentum)
            if close[i] < kama[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA OR RSI > 60 (losing momentum)
            if close[i] > kama[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals