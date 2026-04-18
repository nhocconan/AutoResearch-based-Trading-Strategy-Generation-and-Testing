#!/usr/bin/env python3
"""
4h_RSI_MeanReversion_BollingerBands_TrendFilter
Hypothesis: In ranging markets, price reverts to mean from Bollinger Bands extremes with RSI confirmation.
In trending markets, trade pullbacks to EMA21 in direction of 1d trend. Uses 1d trend filter to adapt
behavior to market regime, working in both bull and bear markets.
Target: 20-40 trades/year on 4h timeframe with disciplined entry conditions.
"""

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
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    ma20 = close_series.rolling(window=20, min_periods=20).mean()
    std20 = close_series.rolling(window=20, min_periods=20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    upper = upper.values
    lower = lower.values
    ma20 = ma20.values
    
    # Calculate RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1-day EMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema21_1d = np.full(len(close_1d), np.nan)
    k = 2 / (21 + 1)
    for i in range(21, len(close_1d)):
        if i == 21:
            ema21_1d[i] = np.mean(close_1d[0:22])
        else:
            ema21_1d[i] = close_1d[i] * k + ema21_1d[i-1] * (1 - k)
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Bollinger Band width for regime detection
    bb_width = (upper - lower) / ma20
    bb_width_ma = np.zeros(n)
    for i in range(20, n):
        bb_width_ma[i] = np.mean(bb_width[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(ma20[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(rsi[i]) or np.isnan(ema21_1d_aligned[i]) or np.isnan(bb_width_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime detection: narrow BB = ranging, wide BB = trending
        is_ranging = bb_width[i] < bb_width_ma[i]
        
        if position == 0:
            if is_ranging:
                # Mean reversion in ranging market
                if close[i] < lower[i] and rsi[i] < 30:
                    signals[i] = 0.25
                    position = 1
                elif close[i] > upper[i] and rsi[i] > 70:
                    signals[i] = -0.25
                    position = -1
            else:
                # Trend following in trending market
                if close[i] > ema21_1d_aligned[i] and close[i] < ma20[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < ema21_1d_aligned[i] and close[i] > ma20[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price reaches middle band or RSI overbought
            if close[i] >= ma20[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches middle band or RSI oversold
            if close[i] <= ma20[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_MeanReversion_BollingerBands_TrendFilter"
timeframe = "4h"
leverage = 1.0