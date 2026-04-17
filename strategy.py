#!/usr/bin/env python3
"""
1h_MultiTimeframe_VolumeBreakout
Strategy: Multi-timeframe volume breakout with ADX trend filter.
Long: 1h price breaks above 4h high AND volume > 2x 20-period avg AND ADX > 25
Short: 1h price breaks below 4h low AND volume > 2x 20-period avg AND ADX > 25
Exit: Price returns to 4h midpoint OR volume drops below threshold
Position size: 0.20
Designed to capture institutional breakouts with volume confirmation across multiple timeframes.
Timeframe: 1h
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
    
    # Calculate volume MA(20) on 1h
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX(14) on 1h for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.insert(tr, 0, 0)
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14 + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14 + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Get 4h data for breakout levels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Align 4h levels to 1h timeframe (wait for 4h bar close)
    high_4h_aligned = align_htf_to_ltf(prices, df_4h, high_4h)
    low_4h_aligned = align_htf_to_ltf(prices, df_4h, low_4h)
    midpoint_4h_aligned = align_htf_to_ltf(prices, df_4h, (high_4h + low_4h) / 2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # volume MA20, ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(high_4h_aligned[i]) or 
            np.isnan(low_4h_aligned[i]) or 
            np.isnan(midpoint_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period average
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx[i] > 25
        
        if position == 0:
            # Long: price breaks above 4h high + volume + trend
            if close[i] > high_4h_aligned[i] and volume_filter and trend_filter:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h low + volume + trend
            elif close[i] < low_4h_aligned[i] and volume_filter and trend_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns to 4h midpoint OR volume drops
            if close[i] < midpoint_4h_aligned[i] or volume[i] < volume_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns to 4h midpoint OR volume drops
            if close[i] > midpoint_4h_aligned[i] or volume[i] < volume_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_MultiTimeframe_VolumeBreakout"
timeframe = "1h"
leverage = 1.0