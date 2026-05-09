#!/usr/bin/env python3
# Hypothesis: 1d timeframe with 1-week Hull Moving Average (HMA) trend filter and 1-day RSI mean reversion.
# In trending markets (price above/below 9-period HMA), RSI extremes signal continuation.
# In ranging markets (price near HMA), RSI extremes signal mean reversion to HMA.
# Uses weekly HMA to avoid whipsaws and capture multi-week trends.
# Entry: Long when RSI < 30 and price > weekly HMA; Short when RSI > 70 and price < weekly HMA.
# Exit: When price crosses back across weekly HMA or RSI returns to neutral (40-60 range).
# Target: 20-60 total trades over 4 years (5-15/year) with size 0.25.

name = "1d_HMA_RSI_MeanReversion"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-week Hull Moving Average (9-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 9:
        return np.zeros(n)
    
    close_1w = df_1w['close']
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    half_len = 9 // 2
    sqrt_len = int(np.sqrt(9))
    
    wma_half = pd.Series(close_1w).rolling(window=half_len, min_periods=half_len).mean()
    wma_full = pd.Series(close_1w).rolling(window=9, min_periods=9).mean()
    raw_hma = 2 * wma_half - wma_full
    hma_9 = pd.Series(raw_hma).rolling(window=sqrt_len, min_periods=sqrt_len).mean()
    hma_9_values = hma_9.values
    hma_9_aligned = align_htf_to_ltf(prices, df_1w, hma_9_values)
    
    # Calculate 1-day RSI (14-period)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(hma_9_aligned[i]) or
            np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI oversold (<30) and price above weekly HMA (uptrend)
            if rsi_values[i] < 30 and close[i] > hma_9_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought (>70) and price below weekly HMA (downtrend)
            elif rsi_values[i] > 70 and close[i] < hma_9_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below weekly HMA OR RSI returns to neutral (>40)
            if close[i] < hma_9_aligned[i] or rsi_values[i] > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly HMA OR RSI returns to neutral (<60)
            if close[i] > hma_9_aligned[i] or rsi_values[i] < 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals