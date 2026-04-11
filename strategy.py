#!/usr/bin/env python3
# 4h_1d_momentum_volume_v1
# Strategy: 4h momentum with volume confirmation and 1-day trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Momentum (ROC > 0) with volume above average and price above 1-day EMA200 indicates strong trend. Long when momentum positive, volume high, and price > 1-day EMA200. Short when momentum negative, volume high, and price < 1-day EMA200. Uses volume filter to avoid false breakouts and trend filter to avoid counter-trend trades. Designed for low frequency (~25-35 trades/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_momentum_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1-day EMA(200) for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 4-hour ROC(12) for momentum (3 periods = 12 hours)
    roc_period = 12
    roc = np.zeros_like(close)
    roc[roc_period:] = (close[roc_period:] - close[:-roc_period]) / close[:-roc_period] * 100
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(50, roc_period, 20), n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(roc[i]) or 
            np.isnan(vol_avg.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: volume above average
        volume_confirm = volume[i] > vol_avg.iloc[i]
        
        # Entry conditions
        if roc[i] > 0 and volume_confirm and close[i] > ema_200_1d_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif roc[i] < 0 and volume_confirm and close[i] < ema_200_1d_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: momentum reversal or trend violation
        elif position == 1 and (roc[i] <= 0 or close[i] < ema_200_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (roc[i] >= 0 or close[i] > ema_200_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals