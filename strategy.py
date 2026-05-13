#!/usr/bin/env python3
"""
1d_1w_RSI_Divergence_Trend
Hypothesis: On 1d timeframe, RSI divergence with price, confirmed by 1w trend,
provides high-probability reversal signals in both bull and bear markets.
RSI divergence indicates weakening momentum before price reverses.
Target: 15-25 trades/year per symbol (60-100 total over 4 years).
"""

name = "1d_1w_RSI_Divergence_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate 14-period RSI
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:13] = 50  # Neutral value for initialization
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w trend: 21 EMA
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    uptrend_1w = close_1w > ema_21_1w
    downtrend_1w = close_1w < ema_21_1w
    
    # Align 1w trend to 1d
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Calculate RSI slope (3-period) and price slope (3-period)
    rsi_slope = np.zeros(n)
    price_slope = np.zeros(n)
    
    for i in range(3, n):
        rsi_slope[i] = rsi[i] - rsi[i-3]
        price_slope[i] = close[i] - close[i-3]
    
    # Detect divergences
    # Bullish divergence: price makes lower low, RSI makes higher low
    bullish_divergence = (price_slope < 0) & (rsi_slope > 0)
    # Bearish divergence: price makes higher high, RSI makes lower high
    bearish_divergence = (price_slope > 0) & (rsi_slope < 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get aligned values
        uptrend = uptrend_1w_aligned[i]
        downtrend = downtrend_1w_aligned[i]
        
        if position == 0:
            # LONG: 1w uptrend + bullish RSI divergence
            if uptrend and bullish_divergence[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: 1w downtrend + bearish RSI divergence
            elif downtrend and bearish_divergence[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 1w trend turns down or bearish divergence appears
            if not uptrend or bearish_divergence[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: 1w trend turns up or bullish divergence appears
            if not downtrend or bullish_divergence[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals