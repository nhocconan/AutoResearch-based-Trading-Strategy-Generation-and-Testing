#!/usr/bin/env python3
"""
6h_MarketFacets_Strategy
Hypothesis: Combine 6h price action with 1d market structure (EMA trend, volume profile, and volatility regime) to capture multi-day moves in BTC/ETH.
Uses 1d EMA34 for trend, 1d volume spike for participation, and 6h ATR-based volatility filter to avoid chop.
Targets 15-25 trades/year by requiring confluence of trend, volume, and volatility conditions.
Works in bull/bear via trend filter + avoids false breakouts in low volatility.
"""

name = "6h_MarketFacets_Strategy"
timeframe = "6h"
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
    
    # Get 1d data for trend, volume, and volatility filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d average volume for volume filter
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate 6h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_6 = pd.Series(tr).rolling(window=6, min_periods=6).mean().values
    atr_avg_6 = pd.Series(atr_6).rolling(window=24, min_periods=24).mean().values  # 4-day average ATR
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA34 (34) and 1d vol avg (20) and 6h ATR avg (24)
    start_idx = max(34, 20, 24)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(atr_avg_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter (1d)
        uptrend_1d = close[i] > ema_34_1d_aligned[i]
        downtrend_1d = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: current 6h volume > 1.5x average 1d volume (scaled)
        vol_6h = volume[i]
        # Scale 1d volume to 6h equivalent (1d = 4x 6h)
        vol_6h_equiv = vol_avg_1d_aligned[i] / 4.0
        volume_filter = vol_6h > vol_6h_equiv * 1.5
        
        # Volatility filter: current ATR > 0.7x average ATR (avoid extremely low volatility)
        vol_filter = atr_6[i] > atr_avg_6[i] * 0.7
        
        if position == 0:
            # Long entry: uptrend + volume participation + adequate volatility
            if uptrend_1d and volume_filter and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + volume participation + adequate volatility
            elif downtrend_1d and volume_filter and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or volatility drops significantly
            if not uptrend_1d or (atr_6[i] < atr_avg_6[i] * 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or volatility drops significantly
            if not downtrend_1d or (atr_6[i] < atr_avg_6[i] * 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals