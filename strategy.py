#!/usr/bin/env python3
"""
12h_1d_ema_trend_pullback_v1
Trend-following strategy using daily EMA for trend direction and 12h EMA pullback entries.
Enters long when price pulls back to 12h EMA during daily uptrend with volume confirmation.
Enters short when price pulls back to 12h EMA during daily downtrend with volume confirmation.
Uses 12h ATR for stop loss. Designed for low trade frequency to minimize fee drag.
"""

name = "12h_1d_ema_trend_pullback_v1"
timeframe = "12h"
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
    volume = prices['volume'].values
    
    # Get daily data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA 50 for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h EMA 21 for pullback entries
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 12h ATR 14 for stop loss
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_21[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price pulls back to EMA21 during daily uptrend with volume
        if (close[i] >= ema_21[i] * 0.995 and close[i] <= ema_21[i] * 1.005 and
            close[i] > ema_50_1d_aligned[i] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price pulls back to EMA21 during daily downtrend with volume
        elif (close[i] >= ema_21[i] * 0.995 and close[i] <= ema_21[i] * 1.005 and
              close[i] < ema_50_1d_aligned[i] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit on opposite EMA cross or ATR stop
        elif position == 1 and (close[i] < ema_50_1d_aligned[i] or 
                                close[i] < ema_21[i] - 2.0 * atr_14[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > ema_50_1d_aligned[i] or 
                                 close[i] > ema_21[i] + 2.0 * atr_14[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals