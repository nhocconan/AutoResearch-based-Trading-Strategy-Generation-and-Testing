#!/usr/bin/env python3
"""
12h_GoldenCross_1dVWAP_Rejection_V1
Hypothesis: Trade the 12h EMA(50) crossing above/below EMA(200) as a long-term trend signal, 
but only enter when price rejects the 1d VWAP (pullback to value area) to avoid chasing extended moves. 
Use volume > 1.5x 24-period average for confirmation. This combines trend-following with mean-reversion 
entries, working in bull markets (riding EMA crosses) and bear markets (avoiding false breaks during 
downtrends by requiring VWAP support/resistance). Targets 15-25 trades/year via infrequent EMA crosses 
+ strict VWAP rejection + volume filter. Uses 1d VWAP for institutional reference and 12h EMA cross 
for trend definition. Works in sideways markets by requiring both trend alignment and value-area entry.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA cross
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # EMA(50) and EMA(200) on 12h
    ema_fast = np.full_like(close_12h, np.nan)
    ema_slow = np.full_like(close_12h, np.nan)
    
    if len(close_12h) >= 50:
        # EMA(50)
        alpha_fast = 2 / (50 + 1)
        ema_fast[0] = close_12h[0]
        for i in range(1, len(close_12h)):
            ema_fast[i] = alpha_fast * close_12h[i] + (1 - alpha_fast) * ema_fast[i-1]
    
    if len(close_12h) >= 200:
        # EMA(200)
        alpha_slow = 2 / (200 + 1)
        ema_slow[0] = close_12h[0]
        for i in range(1, len(close_12h)):
            ema_slow[i] = alpha_slow * close_12h[i] + (1 - alpha_slow) * ema_slow[i-1]
    
    # Align EMA cross to 12h timeframe (same as input, but using alignment for consistency)
    ema_fast_aligned = align_htf_to_ltf(prices, df_12h, ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_12h, ema_slow)
    
    # Get 1d data for VWAP
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d VWAP (typical price * volume)
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    vwap_num = np.full_like(close_1d, np.nan)
    vwap_den = np.full_like(close_1d, np.nan)
    vwap = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 1:
        vwap_num[0] = typical_price[0] * volume_1d[0]
        vwap_den[0] = volume_1d[0]
        vwap[0] = typical_price[0] if volume_1d[0] != 0 else np.nan
        
        for i in range(1, len(close_1d)):
            vwap_num[i] = vwap_num[i-1] + typical_price[i] * volume_1d[i]
            vwap_den[i] = vwap_den[i-1] + volume_1d[i]
            vwap[i] = vwap_num[i] / vwap_den[i] if vwap_den[i] != 0 else np.nan
    
    # Align 1d VWAP to 12h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    
    # Volume confirmation: volume > 1.5x 24-period average (on 12h data)
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, vol_period)  # Need EMA(200) and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_fast_aligned[i]) or np.isnan(ema_slow_aligned[i]) or 
            np.isnan(vwap_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: EMA(50) > EMA(200) + price > VWAP (bullish rejection) + volume
            if ema_fast_aligned[i] > ema_slow_aligned[i] and close[i] > vwap_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: EMA(50) < EMA(200) + price < VWAP (bearish rejection) + volume
            elif ema_fast_aligned[i] < ema_slow_aligned[i] and close[i] < vwap_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: EMA(50) < EMA(200) or price < VWAP (trend break or value rejection)
            if ema_fast_aligned[i] < ema_slow_aligned[i] or close[i] < vwap_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: EMA(50) > EMA(200) or price > VWAP (trend break or value rejection)
            if ema_fast_aligned[i] > ema_slow_aligned[i] or close[i] > vwap_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_GoldenCross_1dVWAP_Rejection_V1"
timeframe = "12h"
leverage = 1.0