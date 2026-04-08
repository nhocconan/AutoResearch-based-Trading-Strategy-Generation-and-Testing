#!/usr/bin/env python3
"""
6h_heiken_ashi_1d_trend_v1
Hypothesis: Heikin Ashi candles smoothed on 1d trend filter for trend continuation entries.
In trending markets, HA candles show consecutive same-color bodies; in ranging markets, they alternate frequently.
Combines trend following with reduced whipsaw vs regular candlesticks. Works in both bull/bear by
only taking trades in direction of higher timeframe trend. Target: 15-35 trades/year on 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_heiken_ashi_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Heikin Ashi candles
    ha_close = (open_price + high + low + close) / 4
    ha_open = np.zeros(n)
    ha_open[0] = (open_price[0] + close[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_high = np.maximum.reduce([high, low, ha_open, ha_close])
    ha_low = np.minimum.reduce([high, low, ha_open, ha_close])
    
    # HA trend: consecutive same-color candles (minimum 3 in a row)
    ha_bullish = ha_close > ha_open
    ha_bearish = ha_close < ha_open
    
    # Count consecutive bullish/bearish candles
    bullish_streak = np.zeros(n, dtype=int)
    bearish_streak = np.zeros(n, dtype=int)
    
    for i in range(1, n):
        if ha_bullish[i]:
            bullish_streak[i] = bullish_streak[i-1] + 1
            bearish_streak[i] = 0
        elif ha_bearish[i]:
            bearish_streak[i] = bearish_streak[i-1] + 1
            bullish_streak[i] = 0
        else:
            bullish_streak[i] = 0
            bearish_streak[i] = 0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if trend filter not ready
        if np.isnan(ema50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: price vs daily EMA50
        above_ema50 = close[i] > ema50_1d_aligned[i]
        below_ema50 = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: trend turns bearish OR HA streak breaks
            if below_ema50 or ha_bearish[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend turns bullish OR HA streak breaks
            if above_ema50 or ha_bullish[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Enter long: bullish HA streak >=3 AND above daily EMA50
            if bullish_streak[i] >= 3 and above_ema50:
                position = 1
                signals[i] = 0.25
            # Enter short: bearish HA streak >=3 AND below daily EMA50
            elif bearish_streak[i] >= 3 and below_ema50:
                position = -1
                signals[i] = -0.25
    
    return signals