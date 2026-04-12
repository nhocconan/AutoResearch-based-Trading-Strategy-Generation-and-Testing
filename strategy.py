#!/usr/bin/env python3
"""
4h_12h_adaptive_ema_crossover
Uses 12h EMA crossover for trend direction (12h EMA20 vs EMA50) and 4h EMA8/EMA21 for entry timing.
Enters long when 12h trend is up and 4h EMA8 crosses above EMA21 with volume confirmation.
Enters short when 12h trend is down and 4h EMA8 crosses below EMA21 with volume confirmation.
Exits when 4h EMA8 crosses back in opposite direction or volume dries up.
Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drift.
Works in trending markets by following higher timeframe trend with lower timeframe precision.
"""

name = "4h_12h_adaptive_ema_crossover"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend direction
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA20 and EMA50 for trend direction
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Trend: 1 = uptrend (EMA20 > EMA50), -1 = downtrend (EMA20 < EMA50), 0 = no trend
    trend_12h = np.where(ema20_12h > ema50_12h, 1, np.where(ema20_12h < ema50_12h, -1, 0))
    
    # Align 12h trend to 4h
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # 4h EMA8 and EMA21 for entry timing
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(trend_12h_aligned[i]) or np.isnan(ema8[i]) or 
            np.isnan(ema21[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            continue
        
        trend = trend_12h_aligned[i]
        
        # Long entry: 12h uptrend and 4h EMA8 crosses above EMA21 with volume
        if trend == 1 and ema8[i] > ema21[i] and ema8[i-1] <= ema21[i-1] and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: 12h downtrend and 4h EMA8 crosses below EMA21 with volume
        elif trend == -1 and ema8[i] < ema21[i] and ema8[i-1] >= ema21[i-1] and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: EMA8 crosses back in opposite direction or volume dries up
        elif position == 1 and (ema8[i] < ema21[i] or not vol_confirm[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (ema8[i] > ema21[i] or not vol_confirm[i]):
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