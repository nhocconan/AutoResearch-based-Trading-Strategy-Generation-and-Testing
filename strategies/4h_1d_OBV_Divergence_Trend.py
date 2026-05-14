#!/usr/bin/env python3
"""
4h_1d_OBV_Divergence_Trend
Hypothesis: On 4h timeframe, On-Balance Volume (OBV) divergence with price, confirmed by 1d trend,
provides high-probability reversal signals in both bull and bear markets. 
OBV divergence indicates weakening momentum before price reverses.
Target: 20-40 trades/year per symbol.
"""

name = "4h_1d_OBV_Divergence_Trend"
timeframe = "4h"
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
    
    # Calculate OBV
    obv = np.zeros(n)
    obv[0] = volume[0]
    for i in range(1, n):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d trend: 50 EMA
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = close_1d > ema_50_1d
    downtrend_1d = close_1d < ema_50_1d
    
    # Align 1d trend to 4h
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Calculate 4h OBV slope (5-period) and price slope (5-period)
    obv_slope = np.zeros(n)
    price_slope = np.zeros(n)
    
    for i in range(5, n):
        obv_slope[i] = obv[i] - obv[i-5]
        price_slope[i] = close[i] - close[i-5]
    
    # Detect divergences
    # Bearish divergence: price makes higher high, OBV makes lower high
    bullish_divergence = (price_slope > 0) & (obv_slope < 0)
    # Bullish divergence: price makes lower low, OBV makes higher low
    bearish_divergence = (price_slope < 0) & (obv_slope > 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get aligned values
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        
        if position == 0:
            # LONG: 1d uptrend + bullish OBV divergence
            if uptrend and bullish_divergence[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + bearish OBV divergence
            elif downtrend and bearish_divergence[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 1d trend turns down or bearish divergence appears
            if not uptrend or bearish_divergence[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: 1d trend turns up or bullish divergence appears
            if not downtrend or bullish_divergence[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals