# 12h_KAMA_Trend_Filter_v1
# Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) on daily timeframe for trend direction,
# combined with 12h price action above/below KAMA and volume confirmation to reduce false signals.
# KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends.
# Targets 15-30 trades/year to minimize fee drag while maintaining trend-following edge.
# Works in both bull and bear markets by adapting smoothing constant to market volatility.

name = "12h_KAMA_Trend_Filter_v1"
timeframe = "12h"
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
    
    # Get daily data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Kaufman Adaptive Moving Average (KAMA)
    # Parameters: ER period=10, Fast SC=2/(2+1)=0.6667, Slow SC=2/(30+1)=0.0645
    er_period = 10
    fast_sc = 2 / (2 + 1)  # 0.6667
    slow_sc = 2 / (30 + 1)  # 0.0645
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0) if len(change) > 0 else np.array([0])
    
    # For proper calculation, we need to compute volatility over er_period window
    volatility_sum = np.zeros_like(close_1d)
    for i in range(er_period, len(close_1d)):
        volatility_sum[i] = np.sum(np.abs(np.diff(close_1d[i-er_period:i+1])))
    
    er = np.zeros_like(close_1d)
    for i in range(er_period, len(close_1d)):
        if volatility_sum[i] > 0:
            er[i] = np.abs(close_1d[i] - close_1d[i-er_period]) / volatility_sum[i]
        else:
            er[i] = 0
    
    # Calculate smoothing constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 12h timeframe
    kama_12h = align_htf_to_ltf(prices, df_1d, kama)
    
    # Volume confirmation: volume > 24-period average (24 * 12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama_12h[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA with volume confirmation
            if close[i] > kama_12h[i] and volume[i] > vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume confirmation
            elif close[i] < kama_12h[i] and volume[i] > vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA
            if close[i] < kama_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above KAMA
            if close[i] > kama_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals