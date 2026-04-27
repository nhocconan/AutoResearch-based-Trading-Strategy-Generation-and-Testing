#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
Breakout at Camarilla R1 (long) / S1 (short) with 4h EMA50 trend filter and volume confirmation.
Only trades during 08-20 UTC to avoid low liquidity periods.
Position size: 0.20.
Target: 15-35 trades/year per symbol.
"""

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
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h_period = 50
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= ema_4h_period:
        ema_4h[ema_4h_period - 1] = np.mean(close_4h[:ema_4h_period])
        for i in range(ema_4h_period, len(close_4h)):
            ema_4h[i] = (close_4h[i] * (2 / (ema_4h_period + 1)) + 
                         ema_4h[i - 1] * (1 - (2 / (ema_4h_period + 1))))
    
    # Align 4h EMA50 to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h volume moving average (20 periods)
    vol_ma = np.full(n, np.nan)
    vol_ma_period = 20
    for i in range(vol_ma_period - 1, n):
        vol_ma[i] = np.mean(volume[i - vol_ma_period + 1:i + 1])
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need volume MA and 4h EMA
    start_idx = max(vol_ma_period - 1, 1)  # need at least 2 bars for prior high/low
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if data not ready
        if np.isnan(vol_ma[i]) or np.isnan(ema_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels from previous bar
        # HLC = (High + Low + Close) / 3
        hlc = (high[i-1] + low[i-1] + close[i-1]) / 3
        range_ = high[i-1] - low[i-1]
        r1 = hlc + (range_ * 1.1 / 12)
        s1 = hlc - (range_ * 1.1 / 12)
        
        # Volume confirmation: current volume > 1.5x average
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume, above 4h EMA (uptrend)
            if close[i] > r1 and volume_ok and close[i] > ema_4h_aligned[i]:
                signals[i] = size
                position = 1
            # Short: price breaks below S1 with volume, below 4h EMA (downtrend)
            elif close[i] < s1 and volume_ok and close[i] < ema_4h_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S1 or trend fails
            if close[i] < s1 or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above R1 or trend fails
            if close[i] > r1 or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0