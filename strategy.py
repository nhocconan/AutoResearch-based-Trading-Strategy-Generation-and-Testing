#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeConfirm_v1
Hypothesis: On 1h timeframe, price breaking Camarilla R1/S1 levels with 4h EMA50 trend filter and volume confirmation captures institutional breakout moves. Using 4h for signal direction avoids lower timeframe noise and overtrading. Target: 60-150 total trades over 4 years (15-37/year) to stay within fee drag limits.
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
    
    # Load 4h data ONCE before loop for HTF trend filter (EMA) and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate Camarilla levels on 4h (based on previous day's OHLC)
    # Camarilla R1 = close + (high - low) * 1.1/12
    # Camarilla S1 = close - (high - low) * 1.1/12
    cam_range = df_4h['high'] - df_4h['low']
    camarilla_r1 = df_4h['close'] + cam_range * 1.1 / 12
    camarilla_s1 = df_4h['close'] - cam_range * 1.1 / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1.values)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1.values)
    
    # Volume spike detection on 1h (volume > 2.0x 24-period EMA = 1 day)
    volume_ema = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    volume_spike = volume > (volume_ema * 2.0)
    
    # Session filter: 08-20 UTC (avoid Asian session noise)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # 4h trend filter (EMA50)
        uptrend = ema_50_aligned[i] > ema_50_aligned[i-1]  # rising EMA
        downtrend = ema_50_aligned[i] < ema_50_aligned[i-1]  # falling EMA
        
        # Long logic: price breaks above Camarilla R1 with volume spike + uptrend
        if close[i] > camarilla_r1_aligned[i] and volume_spike[i] and uptrend:
            if position != 1:
                signals[i] = 0.20
                position = 1
            else:
                signals[i] = 0.20
        # Short logic: price breaks below Camarilla S1 with volume spike + downtrend
        elif close[i] < camarilla_s1_aligned[i] and volume_spike[i] and downtrend:
            if position != -1:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = -0.20
        # Exit conditions: price reaches opposite Camarilla level or trend reversal
        elif position == 1 and (close[i] < camarilla_s1_aligned[i] or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > camarilla_r1_aligned[i] or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0