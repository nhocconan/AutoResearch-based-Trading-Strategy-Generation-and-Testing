#!/usr/bin/env python3
# 1d_KAMA_Trend_1wTrend_Volume
# Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) on daily timeframe for trend direction, filtered by 1-week KAMA trend and volume confirmation.
# KAMA adapts to market noise, reducing false signals in choppy markets while capturing strong trends.
# Works in both bull and bear markets by only trading in the direction of the 1-week trend.
# Target: 7-25 trades/year to stay within optimal frequency range and minimize fee drag.

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
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (14-period) on daily data
    close_series = pd.Series(close)
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series - close_series.shift(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate KAMA (14-period) on 1-week data for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    change_1w = abs(close_1w_series - close_1w_series.shift(10))
    volatility_1w = abs(close_1w_series - close_1w_series.shift(1)).rolling(window=10, min_periods=10).sum()
    er_1w = change_1w / volatility_1w.replace(0, np.nan)
    er_1w = er_1w.fillna(0)
    sc_1w = (er_1w * (0.6645 - 0.0645) + 0.0645) ** 2
    kama_1w = np.zeros(len(close_1w))
    kama_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama_1w[i] = kama_1w[i-1] + sc_1w[i] * (close_1w[i] - kama_1w[i-1])
    
    # Align 1-week KAMA to daily timeframe
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Volume confirmation: volume above 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after sufficient data for indicators
        # Skip if any critical value is NaN
        if (np.isnan(kama[i]) or np.isnan(kama_1w_aligned[i]) or 
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above daily KAMA AND above weekly KAMA AND volume confirmation
            if close[i] > kama[i] and close[i] > kama_1w_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below daily KAMA AND below weekly KAMA AND volume confirmation
            elif close[i] < kama[i] and close[i] < kama_1w_aligned[i] and volume_ok[i]:
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