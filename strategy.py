#!/usr/bin/env python3
# 1d_1w_KAMA_Trend_RSI_Reversal
# Hypothesis: On 1d timeframe, use 1w KAMA trend filter with RSI mean-reversion signals.
# Long when price > 1w KAMA and RSI < 30; short when price < 1w KAMA and RSI > 70.
# Uses 1d volume confirmation (>1.5x 20-day average) to filter false signals.
# Designed for low trade frequency (~15-25 trades/year) to minimize fee drag.
# Works in bull markets via trend-following and bear markets via mean-reversion extremes.

name = "1d_1w_KAMA_Trend_RSI_Reversal"
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
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w KAMA (using close prices)
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    
    # Efficiency Ratio for KAMA
    change = abs(close_1w_series - close_1w_series.shift(10))
    volatility = abs(close_1w_series.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_1w[i] - kama[i-1])
    
    # Align 1w KAMA to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Daily RSI (14-period)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume confirmation (20-day average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 1w KAMA, RSI oversold, volume confirmation
            if (close[i] > kama_aligned[i] and 
                rsi[i] < 30 and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 1w KAMA, RSI overbought, volume confirmation
            elif (close[i] < kama_aligned[i] and 
                  rsi[i] > 70 and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below 1w KAMA or RSI overbought
            if close[i] < kama_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above 1w KAMA or RSI oversold
            if close[i] > kama_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals