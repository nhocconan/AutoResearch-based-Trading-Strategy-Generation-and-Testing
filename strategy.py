#!/usr/bin/env python3
"""
4h_RSI20_Stochastic5_3_Momentum_With_1D_Trend_Filter
Hypothesis: Trade 4h momentum based on RSI(20) and Stochastic(5,3) oversold/overbought levels with 1d EMA trend filter. 
Long when RSI<20 AND Stochastic<10 AND price>1d EMA100; short when RSI>80 AND Stochastic>90 AND price<1d EMA100.
Uses oversold/overbought extremes for mean reversion in range-bound markets while trend filter avoids counter-trend trades.
Target: 80-120 total trades over 4 years (20-30/year) with position size 0.25.
Works in bull/bear: 1d trend filter ensures trades align with higher timeframe direction, RSI/Stochastic extremes provide high-probability mean reversion entries.
"""

name = "4h_RSI20_Stochastic5_3_Momentum_With_1D_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA100 for trend filter
    close_1d = df_1d['close'].values
    ema100_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 100:
        multiplier = 2.0 / (100 + 1)
        ema100_1d[99] = np.mean(close_1d[:100])
        for i in range(100, len(close_1d)):
            ema100_1d[i] = multiplier * close_1d[i] + (1 - multiplier) * ema100_1d[i-1]
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Calculate RSI(20)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First average is simple mean
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = (prev_smoothed * (period-1) + current_value) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = wilder_smooth(gain, 20)
    avg_loss = wilder_smooth(loss, 20)
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Stochastic(5,3)
    # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    lowest_low_5 = np.full_like(low, np.nan)
    highest_high_5 = np.full_like(high, np.nan)
    for i in range(4, n):
        lowest_low_5[i] = np.min(low[i-4:i+1])
        highest_high_5[i] = np.max(high[i-4:i+1])
    
    stoch_k = np.divide((close - lowest_low_5) * 100, (highest_high_5 - lowest_low_5), 
                        out=np.full_like(close, np.nan), where=(highest_high_5 - lowest_low_5)!=0)
    # %D = 3-period SMA of %K
    stoch_d = np.full_like(stoch_k, np.nan)
    for i in range(2, n):
        stoch_d[i] = np.mean(stoch_k[i-2:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(stoch_d[i]) or 
            np.isnan(ema100_1d_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI<20 AND Stochastic<10 AND price>1d EMA100 (oversold in uptrend)
            if rsi[i] < 20 and stoch_d[i] < 10 and close[i] > ema100_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI>80 AND Stochastic>90 AND price<1d EMA100 (overbought in downtrend)
            elif rsi[i] > 80 and stoch_d[i] > 90 and close[i] < ema100_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI>60 OR Stochastic>80 (exit on weakness) OR trend turns down
            if rsi[i] > 60 or stoch_d[i] > 80 or close[i] < ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI<40 OR Stochastic<20 (exit on strength) OR trend turns up
            if rsi[i] < 40 or stoch_d[i] < 20 or close[i] > ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals