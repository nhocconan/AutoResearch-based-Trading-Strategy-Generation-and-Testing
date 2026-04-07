#!/usr/bin/env python3
"""
4h_rolling_std_momentum_12h_trend_volume_v1
Hypothesis: Price momentum relative to rolling standard deviation on 4h combined with 12h trend filter and volume confirmation captures persistent moves in both bull and bear markets. Uses normalized momentum (z-score) to avoid scale issues. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rolling_std_momentum_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = df_12h['close'].ewm(span=50, adjust=False).mean()
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h.values)
    
    # 4h momentum normalized by volatility (z-score of returns)
    returns = np.diff(np.log(close), prepend=0)
    lookback = 20
    mean_return = pd.Series(returns).rolling(window=lookback, min_periods=lookback).mean().values
    std_return = pd.Series(returns).rolling(window=lookback, min_periods=lookback).std().values
    # Avoid division by zero
    std_return = np.where(std_return == 0, 1e-10, std_return)
    z_score = (returns - mean_return) / std_return
    z_score = np.nan_to_num(z_score, nan=0.0)
    
    # Volume confirmation (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if required data not available
        if (np.isnan(z_score[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: momentum turns negative or trend breaks
            if z_score[i] < 0 or close[i] < ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: momentum turns positive or trend breaks
            if z_score[i] > 0 or close[i] > ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: positive momentum, with volume and price above EMA50
            if (z_score[i] > 0.5 and vol_confirm and 
                close[i] > ema_50_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: negative momentum, with volume and price below EMA50
            elif (z_score[i] < -0.5 and vol_confirm and 
                  close[i] < ema_50_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals