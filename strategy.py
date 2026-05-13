#!/usr/bin/env python3
"""
4h_RSI45_Trend_Volume_Spike
Hypothesis: RSI around 45 (neutral) indicates momentum exhaustion and potential continuation of the prevailing trend. Combined with trend filter (4h EMA50) and volume spike (2x 24-bar average), this captures high-probability continuation moves in both bull and bear markets. Designed for low trade frequency (~20-40/year) to minimize fee drag.
"""

name = "4h_RSI45_Trend_Volume_Spike"
timeframe = "4h"
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
    
    # Get 4h data for EMA50 trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA(50) for trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # RSI(14) on 4h close
    delta = np.diff(df_4h['close'], prepend=df_4h['close'][0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi)
    
    # Volume confirmation: current volume > 2.0x 24-period average (4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: RSI near 45 (40-50), above EMA50 (uptrend), volume spike
            if (40 <= rsi_aligned[i] <= 50 and 
                close[i] > ema50_4h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI near 45 (40-50), below EMA50 (downtrend), volume spike
            elif (40 <= rsi_aligned[i] <= 50 and 
                  close[i] < ema50_4h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI moves above 50 (overbought) or below EMA50 (trend change)
            if (rsi_aligned[i] > 50 or 
                close[i] < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI moves below 40 (oversold) or above EMA50 (trend change)
            if (rsi_aligned[i] < 40 or 
                close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals