#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm
Hypothesis: Camarilla R1/S1 breakouts on 12h with 1w EMA50 trend filter and volume confirmation (>1.5x 20-bar avg).
Only trades in direction of weekly trend to avoid counter-trend whipsaws. Uses discrete position sizing (0.25) to limit fee drag.
Designed for low trade frequency (<25/year) to work in both bull and bear markets via trend alignment.
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
    
    # Get 1w data for HTF trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate EMA50 on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels on 1w data (based on previous bar's OHLC)
    camarilla_r1_1w = close_1w + ((high_1w - low_1w) * 1.1 / 12)
    camarilla_s1_1w = close_1w - ((high_1w - low_1w) * 1.1 / 12)
    camarilla_h4_1w = close_1w + ((high_1w - low_1w) * 1.1 / 2)
    camarilla_l4_1w = close_1w - ((high_1w - low_1w) * 1.1 / 2)
    
    # Align HTF indicators to 12h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1_1w, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1_1w, additional_delay_bars=1)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4_1w, additional_delay_bars=1)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4_1w, additional_delay_bars=1)
    
    # Volume confirmation: 1.5x 20-bar average volume (moderate filter)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals with trend filter and volume spike
            # Long: price breaks above R1 in uptrend (close > EMA50) with volume spike
            # Short: price breaks below S1 in downtrend (close < EMA50) with volume spike
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema50_aligned[i]) and volume_spike[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema50_aligned[i]) and volume_spike[i]
            
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
            # Exit when price moves back below Camarilla H4 (take profit at resistance)
            exit_signal = close[i] < camarilla_h4_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla L4 (take profit at support)
            exit_signal = close[i] > camarilla_l4_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0