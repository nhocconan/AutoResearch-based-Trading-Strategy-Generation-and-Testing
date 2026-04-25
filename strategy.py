#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirm_v2
Hypothesis: Trade Camarilla R1/S1 breakouts on 12h with 1d trend filter (price above/below daily EMA50) and volume confirmation (1.5x 28-bar avg). In bullish 1d trend (close > daily EMA50), buy when price breaks above R1; in bearish 1d trend (close < daily EMA50), sell when price breaks below S1. Uses discrete position sizing (0.25) to minimize fee drag and target ~12-25 trades/year. Designed to work in both bull and bear markets by following the higher timeframe trend.
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
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on daily close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 levels
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    camarilla_r1 = close_1d + (1.1 * (high_1d - low_1d) / 12)
    camarilla_s1 = close_1d - (1.1 * (high_1d - low_1d) / 12)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: 1.5x 28-bar average volume (~14 days on 12h chart)
    volume_ma = pd.Series(volume).rolling(window=28, min_periods=28).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 and volume MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend: price above/below daily EMA50
        daily_bullish = close[i] > ema50_1d_aligned[i]
        daily_bearish = close[i] < ema50_1d_aligned[i]
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_r1_aligned[i] and close[i-1] <= camarilla_r1_aligned[i-1]
        breakout_down = close[i] < camarilla_s1_aligned[i] and close[i-1] >= camarilla_s1_aligned[i-1]
        
        if position == 0:
            # Look for breakout signals with volume confirmation and trend alignment
            long_signal = breakout_up and volume_spike[i] and daily_bullish
            short_signal = breakout_down and volume_spike[i] and daily_bearish
            
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
            # Exit when price breaks below S1 or trend changes
            exit_signal = close[i] < camarilla_s1_aligned[i] or not daily_bullish
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price breaks above R1 or trend changes
            exit_signal = close[i] > camarilla_r1_aligned[i] or not daily_bearish
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirm_v2"
timeframe = "12h"
leverage = 1.0