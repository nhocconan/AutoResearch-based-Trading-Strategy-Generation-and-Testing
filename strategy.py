#!/usr/bin/env python3
# 1d_1w_KAMA_Trend_RSI_Momentum
# Hypothesis: Use weekly KAMA trend direction on 1d timeframe with RSI momentum and volume confirmation.
# Long when weekly KAMA up, RSI > 55, and volume above average. Short when weekly KAMA down, RSI < 45, and volume above average.
# Designed for low trade frequency (10-30 trades/year) to minimize fee drag and work in both bull and bear markets by following higher timeframe trend.

name = "1d_1w_KAMA_Trend_RSI_Momentum"
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
    
    # Calculate 1w KAMA for trend
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    
    # Efficiency Ratio
    change = abs(close_1w_series.diff(10))
    volatility = close_1w_series.diff().abs().rolling(window=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0).values
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Align 1w KAMA to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # 1d RSI for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.nan_to_num(rsi, nan=50)
    
    # Volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly KAMA up, RSI bullish, volume confirmation
            if (close[i] > kama_aligned[i] and 
                rsi[i] > 55 and 
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly KAMA down, RSI bearish, volume confirmation
            elif (close[i] < kama_aligned[i] and 
                  rsi[i] < 45 and 
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or RSI overbought
            if close[i] < kama_aligned[i] or rsi[i] > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or RSI oversold
            if close[i] > kama_aligned[i] or rsi[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals