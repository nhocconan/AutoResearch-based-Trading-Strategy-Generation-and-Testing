#!/usr/bin/env python3
"""
4h_1d_SR_Divergence_Trend
Hypothesis: On 4h timeframe, Stochastic RSI divergence with price, confirmed by 1d trend,
provides high-probability reversal signals in both bull and bear markets.
Stochastic RSI captures momentum extremes earlier than RSI, and divergence signals
weakening momentum before price reverses. Target: 20-40 trades/year per symbol.
"""

name = "4h_1d_SR_Divergence_Trend"
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
    
    # Calculate Stochastic RSI (14,3,3,3)
    # First calculate RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic RSI
    stoch_rsi = np.zeros(n)
    for i in range(14, n):
        rsi_window = rsi[i-14:i+1]
        min_rsi = np.min(rsi_window)
        max_rsi = np.max(rsi_window)
        if max_rsi - min_rsi != 0:
            stoch_rsi[i] = (rsi[i] - min_rsi) / (max_rsi - min_rsi) * 100
        else:
            stoch_rsi[i] = 50
    
    # Smooth Stochastic RSI (3-period SMA)
    stoch_rsi_smooth = np.zeros(n)
    for i in range(3, n):
        stoch_rsi_smooth[i] = np.mean(stoch_rsi[i-2:i+1])
    
    # Further smooth (3-period SMA)
    stoch_rsi_final = np.zeros(n)
    for i in range(3, n):
        stoch_rsi_final[i] = np.mean(stoch_rsi_smooth[i-2:i+1])
    
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
    
    # Calculate Stochastic RSI slope (3-period) and price slope (3-period)
    sr_slope = np.zeros(n)
    price_slope = np.zeros(n)
    
    for i in range(3, n):
        sr_slope[i] = stoch_rsi_final[i] - stoch_rsi_final[i-3]
        price_slope[i] = close[i] - close[i-3]
    
    # Detect divergences
    # Bearish divergence: price makes higher high, SR makes lower high
    bearish_divergence = (price_slope > 0) & (sr_slope < 0)
    # Bullish divergence: price makes lower low, SR makes higher low
    bullish_divergence = (price_slope < 0) & (sr_slope > 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get aligned values
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        
        if position == 0:
            # LONG: 1d uptrend + bullish SR divergence
            if uptrend and bullish_divergence[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + bearish SR divergence
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