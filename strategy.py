#!/usr/bin/env python3
"""
4h_DonchianBreakout_Volume_Trend
Hypothesis: Donchian channel breakout with volume confirmation and trend filter on 4h timeframe.
Works in bull/bear via 1d EMA200 trend filter and volatility-adjusted breakouts.
Targets 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
"""

name = "4h_DonchianBreakout_Volume_Trend"
timeframe = "4h"
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
    
    # 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    ema_200 = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean()
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200.values)
    
    # Donchian channels (20-period) on 4h data
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(lookback-1, n):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200)  # Ensure enough data for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of 1d EMA200
        uptrend = close[i] > ema_200_aligned[i]
        downtrend = close[i] < ema_200_aligned[i]
        
        if position == 0:
            # Long: breakout above upper band with volume and uptrend
            if close[i] > upper[i] and volume_confirm[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band with volume and downtrend
            elif close[i] < lower[i] and volume_confirm[i] and downtrend:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower band or trend changes
            if close[i] < lower[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper band or trend changes
            if close[i] > upper[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals