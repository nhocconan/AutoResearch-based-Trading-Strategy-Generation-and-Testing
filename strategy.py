#!/usr/bin/env python3
"""
4h_12h_Pullback_Long_v1
Hypothesis: In strong 12h uptrends (price > 12h EMA50), buy 4h pullbacks to EMA21 with volume confirmation.
Avoids downtrends and ranges. Targets 20-50 trades over 4 years.
Works in bull via trend-following, avoids bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Pullback_Long_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 4h EMA21 for pullback
    ema21_4h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume filter: current volume > 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(ema21_4h[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: 12h price above EMA50
        trend_up = close_12h[i // 3] > ema50_12h[i]  # 12h bar index = i // 3 (4h bars per 12h)
        
        # Pullback condition: price near 4h EMA21 (within 0.5%)
        near_ema = abs(close[i] - ema21_4h[i]) / ema21_4h[i] < 0.005
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma20[i]
        
        # Entry: long on pullback in uptrend with volume
        if trend_up and near_ema and vol_ok and position == 0:
            position = 1
            signals[i] = 0.25
        
        # Exit: trend breakdown or price > 1.5x ATR above EMA21
        elif position == 1:
            # Simple exit: trend breaks down
            if not trend_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
    
    return signals