# 1d_AMA_RSI20_Trend_Filter_Slope
# Hypothesis: Adaptive Moving Average (AMA) adapts to market efficiency, reducing whipsaw in sideways markets while capturing trends.
# Combines AMA slope for trend direction with RSI for momentum filter on daily timeframe.
# Works in bull (trend following) and bear (avoids false signals in chop via adaptive smoothing).
# Target: 20-40 trades/year on 1d timeframe.

#!/usr/bin/env python3
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
    
    # Get 1d data for higher timeframe context (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Adaptive Moving Average (AMA) - Kaufman's Adaptive Moving Average
    # Fast EMA period = 2, Slow EMA period = 30
    fast_sc = 2 / (2 + 1)  # 0.6667
    slow_sc = 2 / (30 + 1)  # 0.0645
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Calculate Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate AMA
    ama = np.zeros_like(close_1d)
    ama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ama[i] = ama[i-1] + sc[i] * (close_1d[i] - ama[i-1])
    
    ama_aligned = align_htf_to_ltf(prices, df_1d, ama)
    
    # Calculate AMA slope (trend direction)
    ama_slope = np.diff(ama, prepend=0)
    ama_slope_aligned = align_htf_to_ltf(prices, df_1d, ama_slope)
    
    # Calculate 1d RSI for momentum filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    # Avoid division by zero
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ama_slope_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: AMA slope positive/negative
        trend_up = ama_slope_aligned[i] > 0
        trend_down = ama_slope_aligned[i] < 0
        
        # Momentum filter: RSI not extreme
        rsi_not_overbought = rsi_aligned[i] < 70
        rsi_not_oversold = rsi_aligned[i] > 30
        
        # Long conditions: up-trend + RSI not overbought
        long_condition = trend_up and rsi_not_overbought
        
        # Short conditions: down-trend + RSI not oversold
        short_condition = trend_down and rsi_not_oversold
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal
        elif position == 1 and not trend_up:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not trend_down:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_AMA_RSI20_Trend_Filter_Slope"
timeframe = "1d"
leverage = 1.0