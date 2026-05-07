#!/usr/bin/env python3
# 1d_KAMA_Trend_1wTrend_Volume
# Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) on daily timeframe for trend direction,
# filtered by 1-week KAMA trend and volume confirmation. KAMA adapts to market noise,
# reducing false signals in choppy markets while capturing trends in both bull and bear markets.
# Target: 15-25 trades/year to stay within optimal frequency range and minimize fee drag.

name = "1d_KAMA_Trend_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (10-period ER, 2 and 30 for smoothing constants) on daily data
    close_daily = close
    change = np.abs(np.diff(close_daily, prepend=close_daily[0]))
    volatility = np.abs(np.diff(close_daily))
    
    # Efficiency Ratio (ER)
    er = np.zeros_like(close_daily)
    er[0] = 0
    for i in range(1, len(close_daily)):
        direction = np.abs(close_daily[i] - close_daily[i-10] if i >= 10 else close_daily[i] - close_daily[0])
        volatility_sum = np.sum(volatility[max(0, i-9):i+1]) if i >= 1 else volatility[i]
        er[i] = direction / volatility_sum if volatility_sum > 0 else 0
    
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close_daily)
    kama[0] = close_daily[0]
    for i in range(1, len(close_daily)):
        kama[i] = kama[i-1] + sc[i] * (close_daily[i] - kama[i-1])
    
    # Calculate KAMA on weekly data for trend filter
    close_1w = df_1w['close'].values
    change_1w = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility_1w = np.abs(np.diff(close_1w))
    
    er_1w = np.zeros_like(close_1w)
    er_1w[0] = 0
    for i in range(1, len(close_1w)):
        direction = np.abs(close_1w[i] - close_1w[i-10] if i >= 10 else close_1w[i] - close_1w[0])
        volatility_sum = np.sum(volatility_1w[max(0, i-9):i+1]) if i >= 1 else volatility_1w[i]
        er_1w[i] = direction / volatility_sum if volatility_sum > 0 else 0
    
    sc_1w = (er_1w * (2/2 - 2/30) + 2/30) ** 2
    
    kama_1w = np.zeros_like(close_1w)
    kama_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama_1w[i] = kama_1w[i-1] + sc_1w[i] * (close_1w[i] - kama_1w[i-1])
    
    # Align weekly KAMA to daily timeframe
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Volume confirmation (20-day average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama[i]) or np.isnan(kama_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above daily KAMA AND above weekly KAMA AND volume spike
            if close[i] > kama[i] and close[i] > kama_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below daily KAMA AND below weekly KAMA AND volume spike
            elif close[i] < kama[i] and close[i] < kama_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price crosses below daily KAMA OR weekly KAMA
            if close[i] < kama[i] or close[i] < kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price crosses above daily KAMA OR weekly KAMA
            if close[i] > kama[i] or close[i] > kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals