#!/usr/bin/env python3
# 4h_momentum_follow_v1
# Hypothesis: Momentum-following strategy using EMA crossover with volume and momentum confirmation.
# Uses EMA(21) and EMA(55) crossover for trend direction, confirmed by volume spike and ROC(10) momentum.
# Designed to work in both bull and bear markets by filtering trades with volume and momentum.
# Target: 20-30 trades/year for low fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_momentum_follow_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend filter (1d EMA200) - load once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily data
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 4h indicators
    # EMA21 and EMA55 for crossover
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema55 = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # ROC(10) for momentum
    roc = np.zeros(n)
    roc[10:] = (close[10:] - close[:-10]) / close[:-10] * 100
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 55  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(ema21[i]) or np.isnan(ema55[i]) or np.isnan(roc[i]) or np.isnan(avg_volume[i]) or np.isnan(ema200_1d_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Daily trend filter
        daily_uptrend = close[i] > ema200_1d_aligned[i]
        daily_downtrend = close[i] < ema200_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: EMA21 crosses below EMA55 or momentum fails
            if ema21[i] < ema55[i] or roc[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA21 crosses above EMA55 or momentum fails
            if ema21[i] > ema55[i] or roc[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Entry conditions
            if volume_ok:
                # Bullish crossover: EMA21 crosses above EMA55 in uptrend with positive momentum
                if ema21[i] > ema55[i] and ema21[i-1] <= ema55[i-1] and daily_uptrend and roc[i] > 0:
                    position = 1
                    signals[i] = 0.25
                # Bearish crossover: EMA21 crosses below EMA55 in downtrend with negative momentum
                elif ema21[i] < ema55[i] and ema21[i-1] >= ema55[i-1] and daily_downtrend and roc[i] < 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals