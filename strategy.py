#!/usr/bin/env python3
# 1D_WeeklyKAMA_Trend_RSI_MeanReversion
# Hypothesis: On daily timeframe, use weekly KAMA to determine trend direction (bull/bear).
# In bull trend (price > weekly KAMA), look for RSI oversold (<30) for long entries.
# In bear trend (price < weekly KAMA), look for RSI overbought (>70) for short entries.
# Exit when RSI returns to neutral zone (40-60). Uses volume confirmation to avoid false signals.
# Designed to work in both bull (mean reversion in uptrend) and bear (mean reversion in downtrend) markets.
# Targets 10-25 trades/year to minimize fee drag on 1d timeframe.

name = "1D_WeeklyKAMA_Trend_RSI_MeanReversion"
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
    if len(df_1w) < 30:  # Need enough data for KAMA
        return np.zeros(n)
    
    # Calculate weekly KAMA (30, 2, 30)
    close_1w = df_1w['close'].values
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Align weekly KAMA to daily
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Daily RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure we have KAMA and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: bull trend (price > weekly KAMA) + RSI oversold + volume
            if (close[i] > kama_aligned[i] and 
                rsi[i] < 30 and 
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: bear trend (price < weekly KAMA) + RSI overbought + volume
            elif (close[i] < kama_aligned[i] and 
                  rsi[i] > 70 and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: RSI returns to neutral zone (40-60)
            if 40 <= rsi[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals