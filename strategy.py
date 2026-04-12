#!/usr/bin/env python3
"""
4h_1d_rsi_pullback_v1
Hypothesis: In strong trends identified by 1-day EMA50, enter on RSI(14) pullbacks to EMA21 on 4h chart with volume confirmation. Works in bull/bear by following the daily trend. Target: 20-40 trades/year (80-160 total) to minimize fee drag.
"""
name = "4h_1d_rsi_pullback_v1"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend direction
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA21 for dynamic support/resistance
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(ema21[i]):
            signals[i] = 0.0
            continue
        
        # Long entry: uptrend (price > 1d EMA50) AND RSI < 40 (pullback) AND price > EMA21 with volume
        if (close[i] > ema50_1d_aligned[i] and rsi[i] < 40 and close[i] > ema21[i] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: downtrend (price < 1d EMA50) AND RSI > 60 (pullback) AND price < EMA21 with volume
        elif (close[i] < ema50_1d_aligned[i] and rsi[i] > 60 and close[i] < ema21[i] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or RSI crosses back to neutral zone (40-60)
        elif position == 1 and (rsi[i] > 60 or close[i] < ema21[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi[i] < 40 or close[i] > ema21[i]):
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