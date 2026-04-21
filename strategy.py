#!/usr/bin/env python3
"""
1h EMA Cross with 4h Trend and Volume Filter
Hypothesis: Use 4h EMA8/EMA21 trend direction for bias, enter on 1h EMA8/EMA21 cross with volume confirmation. This reduces whipsaws in ranging markets while capturing trends. Works in bull/bear by following 4h trend, and volume filter ensures momentum. Target 15-30 trades/year on 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data once for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate EMA8 and EMA21 on 4h
    ema8_4h = pd.Series(close_4h).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    # Align to 1h
    ema8_4h_aligned = align_htf_to_ltf(prices, df_4h, ema8_4h)
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h EMA8/EMA21 for entry timing
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema8_4h_aligned[i]) or np.isnan(ema21_4h_aligned[i]) or 
            np.isnan(ema8[i]) or np.isnan(ema21[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend: bullish if EMA8 > EMA21, bearish if EMA8 < EMA21
        trend_bullish = ema8_4h_aligned[i] > ema21_4h_aligned[i]
        trend_bearish = ema8_4h_aligned[i] < ema21_4h_aligned[i]
        
        # Volume condition
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Enter long only in bullish 4h trend
            if trend_bullish and ema8[i] > ema21[i] and vol_ok:
                signals[i] = 0.20
                position = 1
            # Enter short only in bearish 4h trend
            elif trend_bearish and ema8[i] < ema21[i] and vol_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: EMA cross down OR trend change
            if ema8[i] < ema21[i] or not trend_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: EMA cross up OR trend change
            if ema8[i] > ema21[i] or not trend_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA8_EMA21_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0