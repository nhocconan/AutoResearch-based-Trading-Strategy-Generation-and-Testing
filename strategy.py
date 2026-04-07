#!/usr/bin/env python3
"""
4h_pullback_1d_trend_volume_v1
Hypothesis: Pullback to EMA20 in direction of daily trend with volume confirmation captures high-probability momentum moves.
Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend) by trading with daily trend.
Targets 20-40 trades/year by requiring EMA20 pullback + volume spike + daily trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_pullback_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for EMA20
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Daily data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA20 on 4h close
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    
    # Calculate EMA50 on 1d close
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # 20-period volume average on 4h
    vol_sma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily EMA50 to 4h timeframe (shifted by 1 for completed daily bar)
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema20_4h[i]) or 
            np.isnan(ema50_4h[i]) or 
            np.isnan(vol_sma_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma_4h[i]
        
        if position == 1:  # Long position
            # Exit: price closes below EMA20 OR trend turns down
            if close[i] < ema20_4h[i] or close[i] < ema50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above EMA20 OR trend turns up
            if close[i] > ema20_4h[i] or close[i] > ema50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price closes above EMA20 (pullback long) + volume + uptrend
            if (close[i] > ema20_4h[i] and 
                vol_confirm and 
                close[i] > ema50_4h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price closes below EMA20 (pullback short) + volume + downtrend
            elif (close[i] < ema20_4h[i] and 
                  vol_confirm and 
                  close[i] < ema50_4h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals