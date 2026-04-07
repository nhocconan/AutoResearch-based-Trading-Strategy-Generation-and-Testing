#!/usr/bin/env python3
"""
1h_adaptive_momentum_volume_v1
Hypothesis: Combine momentum (ROC) with volume confirmation and 4h trend filter.
Long when: ROC > 0, volume > 20-period average, and 4h EMA50 trending up.
Short when: ROC < 0, volume > 20-period average, and 4h EMA50 trending down.
Otherwise flat. Uses 1h for timing, 4h for trend direction. Designed for 15-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_adaptive_momentum_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Momentum: Rate of Change over 10 periods
    roc = np.zeros(n)
    roc[10:] = (close[10:] - close[:-10]) / close[:-10]
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h trend filter: EMA50 on 4h timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if outside session
        if not session_mask[i]:
            signals[i] = 0.0
            continue
            
        # Skip if data not available
        if (np.isnan(roc[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or 
            np.isnan(ema_4h_aligned[i])):
            signals[i] = 0.0
            continue
            
        vol_confirmed = volume[i] > vol_ma[i]
        trend_up = close[i] > ema_4h_aligned[i]
        trend_down = close[i] < ema_4h_aligned[i]
        
        # Long: positive momentum + volume + uptrend
        if roc[i] > 0 and vol_confirmed and trend_up:
            signals[i] = 0.20
        # Short: negative momentum + volume + downtrend
        elif roc[i] < 0 and vol_confirmed and trend_down:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals