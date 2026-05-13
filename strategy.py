#!/usr/bin/env python3
"""
1h_Pullback_to_EMA21_TrendFilter_Strategy
Hypothesis: In trending markets (4h EMA50 direction), price pulls back to the 21 EMA on 1h offering high-probability entries. 
Trend filter prevents counter-trend trades, reducing whipsaw in sideways markets. 
Volume confirmation ensures institutional participation. 
Designed for low trade frequency (15-30/year) with strong risk-adjusted returns in both bull and bear markets.
"""

name = "1h_Pullback_to_EMA21_TrendFilter_Strategy"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h EMA50 for trend direction (HTF)
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h EMA21 for pullback entries
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Uptrend (price > 4h EMA50), pullback to 1h EMA21, volume confirmation, session active
            if (ema_50_4h_aligned[i] > 0 and 
                close[i] > ema_50_4h_aligned[i] and
                close[i] <= ema_21[i] * 1.005 and  # Allow 0.5% above EMA21
                close[i] >= ema_21[i] * 0.995 and  # Allow 0.5% below EMA21
                volume_confirm[i] and
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Downtrend (price < 4h EMA50), pullback to 1h EMA21, volume confirmation, session active
            elif (ema_50_4h_aligned[i] > 0 and 
                  close[i] < ema_50_4h_aligned[i] and
                  close[i] <= ema_21[i] * 1.005 and
                  close[i] >= ema_21[i] * 0.995 and
                  volume_confirm[i] and
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 1h EMA21 or 4h trend reverses
            if close[i] < ema_21[i] * 0.995 or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above 1h EMA21 or 4h trend reverses
            if close[i] > ema_21[i] * 1.005 or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals