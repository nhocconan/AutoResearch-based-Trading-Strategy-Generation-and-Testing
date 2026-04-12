#!/usr/bin/env python3
# 4h_1d_ema_cross_v1
# Hypothesis: 4-hour strategy using 1-day EMA cross for trend direction and 1-day price crossing 4-hour SMA for entries, with volume confirmation.
# Works in bull/bear by requiring alignment with the 1d trend (EMA cross) and confirming with volume to avoid false breakouts.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

name = "4h_1d_ema_cross_v1"
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
    
    # Get 1d data for trend and SMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA20 and EMA50 for trend direction
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 4h SMA20 for entry trigger
    sma20_4h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(sma20_4h[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d trend: bullish if EMA20 > EMA50, bearish if EMA20 < EMA50
        bullish_trend = ema20_1d_aligned[i] > ema50_1d_aligned[i]
        bearish_trend = ema20_1d_aligned[i] < ema50_1d_aligned[i]
        
        # Long entry: bullish 1d trend AND price crosses above 4h SMA20 with volume
        if bullish_trend and close[i] > sma20_4h[i] and low[i] <= sma20_4h[i-1] and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: bearish 1d trend AND price crosses below 4h SMA20 with volume
        elif bearish_trend and close[i] < sma20_4h[i] and high[i] >= sma20_4h[i-1] and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: reverse trend or price crosses back through SMA20
        elif position == 1 and (not bullish_trend or close[i] < sma20_4h[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not bearish_trend or close[i] > sma20_4h[i]):
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