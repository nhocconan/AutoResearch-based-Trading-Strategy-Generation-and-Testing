#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeSpike_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h with 12h EMA50 trend filter and volume spike confirmation. In bullish 12h trend, buy breakouts above R1; in bearish 12h trend, sell breakdowns below S1. Volume spike (2.0x 20-bar avg) confirms institutional participation. Designed for 4h timeframe with tight entries (~25-35/year) to minimize fee drag while capturing strong directional moves in both bull and bear markets. Uses discrete position sizing (0.25) to reduce churn.
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
    
    # Get 12h data for HTF trend filter (EMA50) and Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels using previous 12h bar's OHLC
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_r1 = close_12h + 1.1 * (high_12h - low_12h) / 12.0
    camarilla_s1 = close_12h - 1.1 * (high_12h - low_12h) / 12.0
    
    # Align Camarilla levels to 4h timeframe (previous 12h bar's levels available)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # Volume confirmation: 2.0x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50(12h) and volume MA(20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 12h HTF trend
        htf_12h_bullish = close[i] > ema_50_12h_aligned[i]
        htf_12h_bearish = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Look for Camarilla breakouts with volume confirmation
            long_breakout = (high[i] > camarilla_r1_aligned[i]) and volume_spike[i]
            short_breakout = (low[i] < camarilla_s1_aligned[i]) and volume_spike[i]
            
            # Only trade in direction of 12h trend
            if long_breakout and htf_12h_bullish:
                signals[i] = 0.25
                position = 1
            elif short_breakout and htf_12h_bearish:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price returns to Camarilla H3/L3 level or trend reverses
            camarilla_h3 = close_12h + 1.1 * (high_12h - low_12h) / 6.0
            camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
            exit_signal = (low[i] < camarilla_h3_aligned[i]) or (not htf_12h_bullish)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price returns to Camarilla L3/H3 level or trend reverses
            camarilla_l3 = close_12h - 1.1 * (high_12h - low_12h) / 6.0
            camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
            exit_signal = (high[i] > camarilla_l3_aligned[i]) or htf_12h_bullish
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0