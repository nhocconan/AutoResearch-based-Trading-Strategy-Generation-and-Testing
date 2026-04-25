#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h with 1-week EMA50 trend filter and volume confirmation (2.0x 24-bar avg). In bullish 1w trend (close > EMA50), buy when price breaks above R1; in bearish 1w trend (close < EMA50), sell when price breaks below S1. Volume spike confirms participation. Discrete position sizing (0.25) minimizes fee drag. Target ~20-40 trades/year. Designed to work in both bull and bear markets by following the higher timeframe trend.
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
    
    # Get 1w data for HTF trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Camarilla R1 and S1 levels from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla R1 and S1 levels
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    camarilla_range = high_1w - low_1w
    r1 = close_1w + 1.1 * camarilla_range / 12
    s1 = close_1w - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 4h timeframe (1-week lagged)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1, additional_delay_bars=1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1, additional_delay_bars=1)
    
    # Volume confirmation: 2.0x 24-bar average volume (4 days on 4h)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 and volume MA
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend: price above/below EMA50
        trend_bullish = close[i] > ema_50_aligned[i]
        trend_bearish = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Look for breakout signals with volume confirmation and trend alignment
            long_signal = close[i] > r1_aligned[i] and volume_spike[i] and trend_bullish
            short_signal = close[i] < s1_aligned[i] and volume_spike[i] and trend_bearish
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price breaks below S1 or trend reverses
            exit_signal = close[i] < s1_aligned[i] or not trend_bullish
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price breaks above R1 or trend reverses
            exit_signal = close[i] > r1_aligned[i] or not trend_bearish
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0