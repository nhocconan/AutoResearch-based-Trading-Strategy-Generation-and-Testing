#!/usr/bin/env python3
"""
4h_1d_RSI_Pivot_Reversal
Hypothesis: On 4h timeframe, RSI overbought/oversold conditions combined with 1d trend direction
provide high-probability reversal entries. Uses 1d trend filter to avoid counter-trend trades,
and requires RSI divergence for confirmation. Designed to work in both bull and bear markets
by following the higher timeframe trend while capturing short-term reversals.
Target: 25-40 trades/year per symbol.
"""

name = "4h_1d_RSI_Pivot_Reversal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/period)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d trend: 34 EMA (fast) and 89 EMA (slow) for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1d = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    uptrend_1d = ema_34_1d > ema_89_1d
    downtrend_1d = ema_34_1d < ema_89_1d
    
    # Align 1d trend to 4h
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Calculate RSI slope for divergence detection (3-period)
    rsi_slope = np.zeros(n)
    for i in range(3, n):
        rsi_slope[i] = rsi[i] - rsi[i-3]
    
    # Detect divergences
    # Bullish divergence: price makes lower low, RSI makes higher low
    price_down = close < np.roll(close, 1)
    price_down_2bar = (close < np.roll(close, 1)) & (np.roll(close, 1) < np.roll(close, 2))
    rsi_up = rsi > np.roll(rsi, 1)
    bullish_divergence = price_down_2bar & rsi_up & (rsi_slope > 0)
    
    # Bearish divergence: price makes higher high, RSI makes lower high
    price_up = close > np.roll(close, 1)
    price_up_2bar = (close > np.roll(close, 1)) & (np.roll(close, 1) > np.roll(close, 2))
    rsi_down = rsi < np.roll(rsi, 1)
    bearish_divergence = price_up_2bar & rsi_down & (rsi_slope < 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get aligned values
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        
        if position == 0:
            # LONG: 1d uptrend + bullish RSI divergence + RSI < 40 (not overbought)
            if uptrend and bullish_divergence[i] and rsi[i] < 40:
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + bearish RSI divergence + RSI > 60 (not oversold)
            elif downtrend and bearish_divergence[i] and rsi[i] > 60:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 1d trend turns down, RSI > 70 (overbought), or bearish divergence
            if not uptrend or rsi[i] > 70 or bearish_divergence[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: 1d trend turns up, RSI < 30 (oversold), or bullish divergence
            if not downtrend or rsi[i] < 30 or bullish_divergence[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals