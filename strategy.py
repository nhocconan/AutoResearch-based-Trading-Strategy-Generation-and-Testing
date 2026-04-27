#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_New
Hypothesis: 1h Camarilla R1/S1 breakouts with 4h EMA trend filter and volume spike confirmation. 
Uses 4h for signal direction (trend) and 1h for precise entry timing. Added session filter (08-20 UTC) 
to reduce noise trades. Target: 60-150 total trades over 4 years = 15-37/year for 1h. 
Discrete position sizing (0.20) to minimize fee drag. Works in bull/bear markets by following 4h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC) - avoids datetime64 TypeError
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for trend filter
    close_4h_series = pd.Series(df_4h['close'].values)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h OHLC for Camarilla levels
    o_4h = df_4h['open'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Camarilla levels: R1/S1 from 4h OHLC
    camarilla_r1 = c_4h + (h_4h - l_4h) * 1.1 / 12
    camarilla_s1 = c_4h - (h_4h - l_4h) * 1.1 / 12
    
    # Align 4h indicators to 1h timeframe (completed bars only)
    ema_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital (discrete level)
    
    # Warmup: need 4h EMA50 (50) + volume avg (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or \
           np.isnan(ema_aligned[i]) or np.isnan(r1_aligned[i]) or \
           np.isnan(s1_aligned[i]) or np.isnan(volume_confirm[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike
            # Long: price closes above R1 AND above EMA50 (4h uptrend) AND volume spike
            long_condition = (close_val > r1_val) and (close_val > ema_val) and vol_conf
            # Short: price closes below S1 AND below EMA50 (4h downtrend) AND volume spike
            short_condition = (close_val < s1_val) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit: price touches S1 (opposite Camarilla level) or 4h EMA50 turns bearish
            exit_condition = (close_val < s1_val) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price touches R1 (opposite Camarilla level) or 4h EMA50 turns bullish
            exit_condition = (close_val > r1_val) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_New"
timeframe = "1h"
leverage = 1.0