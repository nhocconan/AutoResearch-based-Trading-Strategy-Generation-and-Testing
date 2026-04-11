#!/usr/bin/env python3
# 6h_1d_momentum_reversal_v1
# Strategy: 6h momentum reversal with 1d trend filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: In BTC/ETH, momentum reversals at key daily levels with volume confirmation capture institutional reversals in both bull and bear markets.
# Uses 60-period momentum for trend strength and 1d EMA200 for trend filter to avoid counter-trend trades.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_momentum_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 1d EMA(200) for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 60-period momentum (close / close 60 periods ago - 1)
    momentum = np.zeros_like(close)
    for i in range(60, n):
        if close[i-60] != 0:
            momentum[i] = close[i] / close[i-60] - 1
    
    # Volume confirmation: volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    vol_ratio = pd.Series(volume) / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(momentum[i]) or 
            np.isnan(vol_ratio.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.8x average
        vol_confirmed = vol_ratio.iloc[i] > 1.8
        
        # Entry conditions
        # Long: Negative momentum reversal (oversold) + price above 1d EMA200 (uptrend bias) + volume confirmation
        if vol_confirmed and momentum[i] < -0.05 and close[i] > ema_200_1d_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Positive momentum reversal (overbought) + price below 1d EMA200 (downtrend bias) + volume confirmation
        elif vol_confirmed and momentum[i] > 0.05 and close[i] < ema_200_1d_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: momentum normalization or trend violation
        elif position == 1 and (momentum[i] >= -0.02 or close[i] < ema_200_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (momentum[i] <= 0.02 or close[i] > ema_200_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals