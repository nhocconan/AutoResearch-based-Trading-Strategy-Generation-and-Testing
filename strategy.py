#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolFilter_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 1h with 4h EMA50 trend filter and 1d volume spike confirmation. 
In bullish 4h trend (close > EMA50), buy when price breaks above R1; in bearish 4h trend (close < EMA50), 
sell when price breaks below S1. Volume spike (2.0x 24-bar average on 1d) confirms participation. 
Uses discrete position sizing (0.20) to minimize fee drag and target ~20-40 trades/year. 
Designed to work in both bull and bear markets by following the 4h trend, with 1h for precise entry timing.
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
    
    # Get 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close for trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Get 1d data for Camarilla levels and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla R1 and S1 levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 levels
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    camarilla_range = high_1d - low_1d
    r1 = close_1d + 1.1 * camarilla_range / 12
    s1 = close_1d - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 1h timeframe (1-day lagged)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1, additional_delay_bars=1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1, additional_delay_bars=1)
    
    # Volume confirmation: 2.0x 24-bar average volume on 1d (requires full day history)
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=24, min_periods=24).mean().values
    volume_spike_1d = volume_1d > (2.0 * volume_ma_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50 bars on 4h) and volume MA (24 bars on 1d)
    # Convert to 1h approximate: 50*4 = 200 bars for EMA50, 24*4 = 96 bars for volume MA
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine 4h HTF trend: price above/below EMA50
        trend_bullish = close[i] > ema_50_aligned[i]
        trend_bearish = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Look for breakout signals with volume confirmation and trend alignment
            long_signal = close[i] > r1_aligned[i] and volume_spike_aligned[i] and trend_bullish
            short_signal = close[i] < s1_aligned[i] and volume_spike_aligned[i] and trend_bearish
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit when price breaks below S1 or trend reverses
            exit_signal = close[i] < s1_aligned[i] or not trend_bullish
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit when price breaks above R1 or trend reverses
            exit_signal = close[i] > r1_aligned[i] or not trend_bearish
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolFilter_v1"
timeframe = "1h"
leverage = 1.0