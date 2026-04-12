#!/usr/bin/env python3
# 12h_1w_ema_trend_with_volume_and_atr
# Hypothesis: 12-hour trend following using weekly EMA trend filter, with price crossing above/below EMA as entry signal.
# Volume confirmation and ATR volatility filter reduce false signals. Works in bull/bear by only taking trades in direction of higher timeframe trend.
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.

name = "12h_1w_ema_trend_with_volume_and_atr"
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
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get daily data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily ATR(14) for volatility filter
    tr1 = np.abs(np.subtract(high_1d, low_1d))
    tr2 = np.abs(np.subtract(high_1d, np.roll(close_1d, 1)))
    tr3 = np.abs(np.subtract(low_1d, np.roll(close_1d, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly EMA and daily ATR to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Volume confirmation: volume > 1.3x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price crosses above weekly EMA with volume and volatility filter
        if (close[i] > ema_50_1w_aligned[i] and vol_confirm[i] and 
            atr_aligned[i] > 0 and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price crosses below weekly EMA with volume and volatility filter
        elif (close[i] < ema_50_1w_aligned[i] and vol_confirm[i] and 
              atr_aligned[i] > 0 and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal
        elif position == 1 and close[i] < ema_50_1w_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema_50_1w_aligned[i]:
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