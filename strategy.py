#!/usr/bin/env python3
"""
1d_1w_Weekly_Pullback_Long_Short
Hypothesis: Trade pullbacks in weekly trend using 1d price action. In bull markets (price above weekly EMA20), go long on 1d pullbacks to EMA20 with volume confirmation. In bear markets (price below weekly EMA20), go short on 1d bounces to EMA20 with volume confirmation. Uses weekly EMA for trend filter and 1d EMA20 for pullback target, reducing whipsaws. Targets 10-20 trades/year by requiring trend alignment and pullback to EMA20 with volume > 1.5x 20-day average. Works in bull markets by buying dips in uptrend, and in bear markets by selling rallies in downtrend.
"""

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False).values
    
    # Align weekly EMA to daily
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily EMA20 for pullback target
    ema_20 = pd.Series(close).ewm(span=20, adjust=False).values
    
    # Volume confirmation: current volume > 1.5 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need EMA20 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(ema_20[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: bullish trend (price > weekly EMA20) and pullback to daily EMA20 with volume
            if (close[i] > ema_1w_aligned[i] and 
                low[i] <= ema_20[i] <= high[i] and  # price touches EMA20
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish trend (price < weekly EMA20) and bounce to daily EMA20 with volume
            elif (close[i] < ema_1w_aligned[i] and 
                  low[i] <= ema_20[i] <= high[i] and  # price touches EMA20
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price closes below EMA20 or trend turns bearish
            if close[i] < ema_20[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above EMA20 or trend turns bullish
            if close[i] > ema_20[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Weekly_Pullback_Long_Short"
timeframe = "1d"
leverage = 1.0