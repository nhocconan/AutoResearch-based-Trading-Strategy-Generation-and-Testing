#!/usr/bin/env python3
"""
1h_Camarilla_H3L3_Breakout_4hTrendFilter_VolumeConfirm_v1
Hypothesis: Trade Camarilla H3/L3 breakouts on 1h with 4h EMA50 trend filter and volume confirmation.
Camarilla pivot levels identify intraday support/resistance; breakouts above H3 or below L3 with
4h trend alignment and volume > 1.5x 20-bar average capture momentum moves in both bull and bear markets.
The 4h trend filter avoids counter-trend trades, reducing whipsaw in ranging/choppy conditions.
Target: 15-37 trades/year per symbol (60-150 total over 4 years).
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
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for HTF trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate daily Camarilla pivot levels (H3, L3, H4, L4)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    H3 = pivot + range_1d * 1.1 / 4
    L3 = pivot - range_1d * 1.1 / 4
    H4 = pivot + range_1d * 1.1 / 2
    L4 = pivot - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume confirmation: volume > 1.5x 20-bar SMA
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_sma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume SMA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine 4h HTF trend (bullish = price above EMA50)
        htf_4h_bullish = close[i] > ema_50_4h_aligned[i]
        htf_4h_bearish = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above H3 + 4h uptrend + volume + session
            long_setup = (close[i] > H3_aligned[i]) and htf_4h_bullish and volume_confirm[i] and session_filter[i]
            
            # Short setup: price breaks below L3 + 4h downtrend + volume + session
            short_setup = (close[i] < L3_aligned[i]) and htf_4h_bearish and volume_confirm[i] and session_filter[i]
            
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
            # Exit: price touches L3 (stop) OR 4h trend turns bearish
            if (close[i] <= L3_aligned[i]) or (not htf_4h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price touches H3 (stop) OR 4h trend turns bullish
            if (close[i] >= H3_aligned[i]) or (htf_4h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hTrendFilter_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0