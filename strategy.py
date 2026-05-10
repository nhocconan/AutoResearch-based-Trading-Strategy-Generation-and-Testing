#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeFilter
# Hypothesis: Use 4h for trend direction (EMA50), daily for Camarilla pivot levels (R1/S1), and 1h for entry timing with volume confirmation.
# Only trade during 08-20 UTC session to reduce noise. Target 15-37 trades/year by requiring confluence of trend, level break, and volume spike.
# Works in bull/bear via trend filter and volatility-adjusted volume confirmation.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeFilter"
timeframe = "1h"
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
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate daily typical price and range for Camarilla (R1/S1)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    
    # Camarilla R1 and S1 levels (using standard formula)
    r1 = typical_price + range_hl * 1.083 / 2
    s1 = typical_price - range_hl * 1.083 / 2
    
    # Align Camarilla levels to 1h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # Volume confirmation (24-period MA on 1h chart, ~1 day)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 4h EMA50 (50), daily Camarilla (need at least 1 day), volume MA (24)
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 4h trend filter
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation (1.5x average volume)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Breakout conditions
        breakout_long = close[i] > r1_aligned[i]
        breakout_short = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R1 + 4h uptrend + volume spike
            if breakout_long and uptrend and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S1 + 4h downtrend + volume spike
            elif breakout_short and downtrend and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks back below R1 or 4h trend turns down
            if close[i] < r1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks back above S1 or 4h trend turns up
            if close[i] > s1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals