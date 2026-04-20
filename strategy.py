#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 12h EMA200 Trend Filter
# Williams %R identifies overbought/oversold conditions on 6h chart.
# Only take long signals when price is above 12h EMA200 (bullish trend bias).
# Only take short signals when price is below 12h EMA200 (bearish trend bias).
# This avoids counter-trend trades in strong trends, reducing whipsaw.
# Williams %R uses 14-period lookback: oversold < -80, overbought > -20.
# Designed to work in both bull and bear markets by aligning with higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 200-period EMA on 12h timeframe
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Calculate Williams %R on 6h chart (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    willr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if NaN in indicators
        if np.isnan(willr[i]) or np.isnan(ema200_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema200 = ema200_12h_aligned[i]
        wr = willr[i]
        
        if position == 0:
            # Enter long: oversold + bullish trend (price above EMA200)
            if wr < -80 and price > ema200:
                signals[i] = 0.25
                position = 1
            # Enter short: overbought + bearish trend (price below EMA200)
            elif wr > -20 and price < ema200:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: either overbought or trend turns bearish
            if wr > -20 or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: either oversold or trend turns bullish
            if wr < -80 or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_EMA200_TrendFilter"
timeframe = "6h"
leverage = 1.0