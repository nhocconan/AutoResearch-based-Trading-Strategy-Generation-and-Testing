#!/usr/bin/env python3
"""
1h_4hTrend_1dVolume_CamPullback
Hypothesis: 1h timeframe with 4h trend filter (EMA20) and 1d volume confirmation.
Enters on pullback to 4h EMA20 during 4h trend (pullback in opposite direction of trend).
Uses 1d volume spike (>1.5x 20-day average) for confirmation.
Designed for 15-30 trades/year to avoid fee drag in 1h.
Works in bull/bear via trend filter and volume confirmation.
"""

name = "1h_4hTrend_1dVolume_CamPullback"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h data for trend and pullback logic
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA20 for trend
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volume average
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.divide(vol_1d, vol_ma20_1d, out=np.zeros_like(vol_1d), where=vol_ma20_1d!=0)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend
        close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
        if np.isnan(close_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = close_4h_aligned[i] > ema_20_4h_aligned[i]
        trend_down = close_4h_aligned[i] < ema_20_4h_aligned[i]
        
        if position == 0:
            # Long: pullback to EMA20 during uptrend with volume confirmation
            if (low[i] <= ema_20_4h_aligned[i] and 
                close[i] > ema_20_4h_aligned[i] and  # closed back above EMA
                vol_ratio_1d_aligned[i] > 1.5 and 
                trend_up):
                signals[i] = 0.20
                position = 1
            # Short: pullback to EMA20 during downtrend with volume confirmation
            elif (high[i] >= ema_20_4h_aligned[i] and 
                  close[i] < ema_20_4h_aligned[i] and  # closed back below EMA
                  vol_ratio_1d_aligned[i] > 1.5 and 
                  trend_down):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below EMA20 or trend turns down
            if (close[i] < ema_20_4h_aligned[i] or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above EMA20 or trend turns up
            if (close[i] > ema_20_4h_aligned[i] or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals