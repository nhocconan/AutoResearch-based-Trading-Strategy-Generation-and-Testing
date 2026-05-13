#!/usr/bin/env python3
"""
4h_Trend_Pullback_Breakout
Hypothesis: Combine daily trend filter with 4-hour pullback entries. Long when 1d trend is up (close > EMA50), price pulls back to EMA21 on 4h and breaks above swing high, with volume confirmation. Short when 1d trend is down (close < EMA50), price pulls back to EMA21 and breaks below swing low, with volume confirmation. Uses ATR stop loss to manage risk. Designed for 4h timeframe to capture swings in both bull and bear markets with limited trades.
"""

name = "4h_Trend_Pullback_Breakout"
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
    volume = prices['volume'].values
    
    # 4h indicators
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Swing points: 5-period high/low
    swing_high = pd.Series(high).rolling(window=5, min_periods=5).max().shift(1).values
    swing_low = pd.Series(low).rolling(window=5, min_periods=5).min().shift(1).values
    
    # Volume average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: 1d uptrend, pullback to EMA21, break above swing high, volume spike
            if (uptrend_1d_aligned[i] and 
                close[i] <= ema_21[i] * 1.01 and  # near EMA21 (within 1%)
                close[i] > swing_high[i] and 
                volume[i] > vol_avg[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend, pullback to EMA21, break below swing low, volume spike
            elif (downtrend_1d_aligned[i] and 
                  close[i] >= ema_21[i] * 0.99 and  # near EMA21 (within 1%)
                  close[i] < swing_low[i] and 
                  volume[i] > vol_avg[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close below EMA21 or trend change
            if close[i] < ema_21[i] * 0.99 or not uptrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: close above EMA21 or trend change
            if close[i] > ema_21[i] * 1.01 or not downtrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals