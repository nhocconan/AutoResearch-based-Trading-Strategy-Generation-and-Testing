#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter + 1-day RSI mean reversion
# Long when 1-day RSI < 30 AND Choppiness Index > 61.8 (range regime)
# Short when 1-day RSI > 70 AND Choppiness Index > 61.8 (range regime)
# Exit when RSI crosses back to neutral (40-60 range)
# Uses regime filter to avoid trending markets where mean reversion fails
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load daily data ONCE before loop for RSI
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day RSI (14-period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate Choppiness Index on 4h (14-period)
    atr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr[0] = high[0] - low[0]  # first value
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (highest_high14 - lowest_low14)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 14
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        rsi = rsi_1d_aligned[i]
        chop_val = chop[i]
        
        if position == 0:
            # Long setup: RSI oversold AND choppy market (range regime)
            if rsi < 30 and chop_val > 61.8:
                position = 1
                signals[i] = position_size
            # Short setup: RSI overbought AND choppy market (range regime)
            elif rsi > 70 and chop_val > 61.8:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral range (40-60)
            if rsi >= 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral range (40-60)
            if rsi <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Chop_RSI_MeanReversion"
timeframe = "4h"
leverage = 1.0