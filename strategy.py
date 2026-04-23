#!/usr/bin/env python3
"""
Hypothesis: 6h Volume-Weighted RSI with Weekly Trend Filter
- Uses 6h RSI(14) with volume weighting to identify momentum exhaustion
- Weekly EMA20 defines higher timeframe trend: only trade pullbacks in trend direction
- Volume-weighted RSI reduces false signals during low-volume chop
- Targets 12-37 trades/year by requiring both momentum extreme and volume confirmation
- Works in bull/bear markets by trading pullbacks with the weekly trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate volume-weighted RSI(14) on 6h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Volume-weighted gain/loss
    vol_weight = volume / (np.mean(volume) + 1e-10)
    vol_weight = np.clip(vol_weight, 0.5, 2.0)  # cap extreme volume weights
    
    weighted_gain = gain * vol_weight
    weighted_loss = loss * vol_weight
    
    # Wilder's smoothing with volume weighting
    avg_gain = pd.Series(weighted_gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(weighted_loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (< 30) AND price above weekly EMA20
            if (rsi[i] < 30 and close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (> 70) AND price below weekly EMA20
            elif (rsi[i] > 70 and close[i] < ema_20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI returns to neutral (40-60) OR trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long when RSI >= 40 OR price closes below weekly EMA20
                if (rsi[i] >= 40 or close[i] < ema_20_1w_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short when RSI <= 60 OR price closes above weekly EMA20
                if (rsi[i] <= 60 or close[i] > ema_20_1w_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_VolumeWeightedRSI_WeeklyEMA20_Trend"
timeframe = "6h"
leverage = 1.0