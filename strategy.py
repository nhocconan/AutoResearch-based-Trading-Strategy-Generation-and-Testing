#!/usr/bin/env python3
# 12h_1d_roc_volume_momentum_v1
# Strategy: 12h Rate-of-Change with volume confirmation and 1d trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: ROC captures momentum bursts. Volume > 1.5x 20-period average confirms institutional participation.
# 1d EMA50 filter ensures we only trade in the direction of the higher timeframe trend.
# Designed for low trade frequency (~15-30/year) to minimize fee drift.
# Works in bull markets via momentum continuation and bear markets via short signals during distribution.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_roc_volume_momentum_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h ROC(10) for momentum
    roc = np.zeros_like(close)
    roc[10:] = (close[10:] - close[:-10]) / close[:-10] * 100
    
    # 12h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(roc[i]) or np.isnan(vol_avg_20[i]) or np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # ROC momentum: >0 for bullish, <0 for bearish
        roc_bullish = roc[i] > 0
        roc_bearish = roc[i] < 0
        
        # 1d trend filter: price above/below EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: ROC bullish AND volume confirmation AND 1d uptrend
        if roc_bullish and vol_confirm and trend_up and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: ROC bearish AND volume confirmation AND 1d downtrend
        elif roc_bearish and vol_confirm and trend_down and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite ROC signal (momentum fade)
        elif position == 1 and roc_bearish:
            position = 0
            signals[i] = 0.0
        elif position == -1 and roc_bullish:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals