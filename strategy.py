#!/usr/bin/env python3
# 4h_ThreeBarPattern_1dTrend_VolumeFilter
# Hypothesis: Three-bar momentum pattern (close > previous high or close < previous low) with 1d EMA trend filter and volume spike.
# Works in bull/bear by trading with daily trend direction only. Pattern identifies short-term momentum bursts.
# Targets 20-40 trades/year to minimize fee drag while capturing explosive moves.

name = "4h_ThreeBarPattern_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20) + 1  # Warmup for daily EMA + volume MA + 1 bar for pattern
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(high[i-1]) or np.isnan(low[i-1])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Three-bar momentum pattern
        # Bullish: current close breaks above previous bar's high
        bullish_pattern = close[i] > high[i-1]
        # Bearish: current close breaks below previous bar's low
        bearish_pattern = close[i] < low[i-1]
        
        if position == 0:
            # Long entry: bullish pattern + uptrend + volume spike
            if bullish_pattern and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish pattern + downtrend + volume spike
            elif bearish_pattern and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: pattern fails or trend reverses
            if not bullish_pattern or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: pattern fails or trend reverses
            if not bearish_pattern or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals