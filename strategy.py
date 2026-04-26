#!/usr/bin/env python3
"""
1d_Camilla_R3S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 levels act as strong intraday support/resistance on 1d timeframe. 
Breakout above R3 with volume spike and weekly uptrend = long. Breakdown below S3 with volume 
spike and weekly downtrend = short. Uses weekly trend filter to avoid counter-trend trades. 
Target: 7-25 trades/year per symbol. Timeframe: 1d, HTF: 1w
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
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter (more responsive than EMA50)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla pivot levels for previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3
    # Range = high - low
    daily_range = high - low
    
    # Camarilla levels (based on previous day's data)
    # R4 = close + range * 1.1/2
    # R3 = close + range * 1.1/4
    # R2 = close + range * 1.1/6
    # R1 = close + range * 1.1/12
    # PP = (high + low + close) / 3
    # S1 = close - range * 1.1/12
    # S2 = close - range * 1.1/6
    # S3 = close - range * 1.1/4
    # S4 = close - range * 1.1/2
    
    # Use previous day's data to calculate today's levels (avoid look-ahead)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First bar has no previous data
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    prev_typical = (prev_high + prev_low + prev_close) / 3
    prev_range = prev_high - prev_low
    
    # Calculate Camarilla levels
    R3 = prev_close + prev_range * 1.1 / 4
    S3 = prev_close - prev_range * 1.1 / 4
    
    # Volume spike detector (20-period volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter from weekly EMA34
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 with volume spike and weekly uptrend
            if close[i] > R3[i] and volume_spike[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume spike and weekly downtrend
            elif close[i] < S3[i] and volume_spike[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price re-enters below R3 OR weekly trend changes to downtrend
            if close[i] < R3[i] or not uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price re-enters above S3 OR weekly trend changes to uptrend
            if close[i] > S3[i] or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0