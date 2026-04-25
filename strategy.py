#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolFilter
Hypothesis: Trade 1h Camarilla R1/S1 breakouts aligned with 4h trend direction (EMA50) and confirmed by 1d volume spike (2.0x 20-bar avg). Uses discrete position sizing (0.20) to limit fee drag and targets 60-150 total trades over 4 years. Designed to work in both bull and bear markets by following 4h trend while avoiding low-volume false breakouts.
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
    
    # Get 4h data for HTF trend (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d volume MA(20) for spike detection
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate Camarilla levels using previous 1h bar's OHLC (for 1h timeframe)
    # We need to shift 1h OHLC by 1 bar to use previous bar's levels
    high_1h_shift = np.roll(high, 1)
    low_1h_shift = np.roll(low, 1)
    close_1h_shift = np.roll(close, 1)
    # First bar will have invalid shifted values (will be filtered by min_periods later)
    camarilla_r1 = close_1h_shift + 1.1 * (high_1h_shift - low_1h_shift) / 12.0
    camarilla_s1 = close_1h_shift - 1.1 * (high_1h_shift - low_1h_shift) / 12.0
    
    # Align Camarilla levels to 1h (they are already 1h-aligned via shift)
    camarilla_r1_aligned = camarilla_r1
    camarilla_s1_aligned = camarilla_s1
    
    # Volume confirmation: 2.0x 20-bar average volume on 1d
    volume_spike = volume > (2.0 * volume_ma_20_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50(4h) and volume MA(20) on 1d
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready (NaN from alignment or rolling)
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine 4h HTF trend
        htf_4h_bullish = close[i] > ema_50_4h_aligned[i]
        htf_4h_bearish = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Look for Camarilla breakouts with volume confirmation
            long_breakout = (high[i] > camarilla_r1_aligned[i]) and volume_spike[i]
            short_breakout = (low[i] < camarilla_s1_aligned[i]) and volume_spike[i]
            
            # Only trade in direction of 4h trend
            if long_breakout and htf_4h_bullish:
                signals[i] = 0.20
                position = 1
            elif short_breakout and htf_4h_bearish:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit when price returns to Camarilla H3/L3 level or trend reverses
            camarilla_h3 = close_1h_shift + 1.1 * (high_1h_shift - low_1h_shift) / 6.0
            camarilla_h3_aligned = camarilla_h3  # already 1h-aligned
            exit_signal = (low[i] < camarilla_h3_aligned[i]) or (not htf_4h_bullish)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit when price returns to Camarilla L3/H3 level or trend reverses
            camarilla_l3 = close_1h_shift - 1.1 * (high_1h_shift - low_1h_shift) / 6.0
            camarilla_l3_aligned = camarilla_l3  # already 1h-aligned
            exit_signal = (high[i] > camarilla_l3_aligned[i]) or htf_4h_bullish
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolFilter"
timeframe = "1h"
leverage = 1.0