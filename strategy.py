#!/usr/bin/env python3
"""
6h_RSI_Extreme_Trend_Filter
Hypothesis: Use 6-hour RSI extremes (above 70 or below 30) to identify overbought/oversold conditions, but only trade in the direction of the weekly trend (using weekly EMA50) to avoid counter-trend trades. This strategy aims to capture mean-reversion bounces within a strong trend, which works in both bull and bear markets. The weekly trend filter reduces false signals during trend exhaustion, while RSI extremes provide precise entry timing. Designed for 6h timeframe to limit trades (target: 15-35/year) and avoid fee drag.
"""

name = "6h_RSI_Extreme_Trend_Filter"
timeframe = "6h"
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
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate RSI (14-period) on 6h data
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if any required data is NaN
        if np.isnan(rsi[i]) or np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI below 30 (oversold) AND price above weekly EMA50 (uptrend)
            if rsi[i] < 30 and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI above 70 (overbought) AND price below weekly EMA50 (downtrend)
            elif rsi[i] > 70 and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses above 50 (momentum shift) or trend change
            if rsi[i] > 50 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses below 50 (momentum shift) or trend change
            if rsi[i] < 50 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals