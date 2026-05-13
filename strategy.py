#!/usr/bin/env python3
"""
1d_1w_RSI_Divergence_Signal
Hypothesis: RSI divergence on daily timeframe with weekly trend filter provides high-probability reversal signals. RSI divergence signals weakening momentum before price reverses. Weekly trend filter ensures alignment with higher timeframe momentum, reducing false signals in chop. Designed for low trade frequency to minimize fee drag in both bull and bear markets.
"""

name = "1d_1w_RSI_Divergence_Signal"
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
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (equivalent to EMA with alpha=1/14)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])  # First average of first 14 periods
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly trend: 34 EMA
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1w = close_1w > ema_34_1w
    downtrend_1w = close_1w < ema_34_1w
    
    # Align weekly trend to daily
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Calculate RSI slope (5-period) and price slope (5-period)
    rsi_slope = np.zeros(n)
    price_slope = np.zeros(n)
    
    for i in range(5, n):
        rsi_slope[i] = rsi[i] - rsi[i-5]
        price_slope[i] = close[i] - close[i-5]
    
    # Detect divergences
    # Bearish divergence: price makes higher high, RSI makes lower high
    bearish_divergence = (price_slope > 0) & (rsi_slope < 0)
    # Bullish divergence: price makes lower low, RSI makes higher low
    bullish_divergence = (price_slope < 0) & (rsi_slope > 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get aligned values
        uptrend = uptrend_1w_aligned[i]
        downtrend = downtrend_1w_aligned[i]
        
        if position == 0:
            # LONG: weekly uptrend + bullish RSI divergence
            if uptrend and bullish_divergence[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: weekly downtrend + bearish RSI divergence
            elif downtrend and bearish_divergence[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: weekly trend turns down or bearish divergence appears
            if not uptrend or bearish_divergence[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: weekly trend turns up or bullish divergence appears
            if not downtrend or bullish_divergence[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals