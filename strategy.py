#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrendFilter_VolumeSpike_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 1h with 4h EMA34 trend filter and volume spike confirmation. 
Camarilla R1/S1 levels provide tight reversal/breakout points. In strong 4h trends (price above/below EMA34), 
breakouts of R1 (resistance) or S1 (support) continue the trend. Volume spike confirms institutional participation. 
Session filter (08-20 UTC) reduces noise. Discrete sizing (0.20) limits fee drift. Target: 15-35 trades/year per symbol.
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
    
    # Get 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for HTF trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data for Camarilla pivot calculation (yesterday's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1/S1, R2/S2 (using close of previous day)
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma[i]) or not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine 4h HTF trend (bullish = price above EMA34)
        htf_4h_bullish = close[i] > ema_34_4h_aligned[i]
        htf_4h_bearish = close[i] < ema_34_4h_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above R1 + 4h uptrend + volume spike
            long_setup = (close[i] > camarilla_r1_aligned[i]) and htf_4h_bullish and volume_spike[i]
            
            # Short setup: price breaks below S1 + 4h downtrend + volume spike
            short_setup = (close[i] < camarilla_s1_aligned[i]) and htf_4h_bearish and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price touches S1 (stop) OR 4h trend turns bearish
            if (close[i] <= camarilla_s1_aligned[i]) or (not htf_4h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price touches R1 (stop) OR 4h trend turns bullish
            if (close[i] >= camarilla_r1_aligned[i]) or (htf_4h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrendFilter_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0