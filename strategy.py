#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeFilter_v2
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h with 1d EMA200 trend filter and volume confirmation. In bullish 1d trend (price > EMA200), buy breakouts above R1; in bearish 1d trend (price < EMA200), sell breakdowns below S1. Volume spike (1.8x 20-bar avg) confirms participation. Uses discrete position sizing (0.25) to minimize fee drag and target ~25-35 trades/year. Designed to work in both bull and bear markets by following the higher timeframe trend.
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
    
    # Get 1d data for HTF trend filter (EMA200) and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Camarilla levels using previous 1d bar's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12.0
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12.0
    
    # Align Camarilla levels to 4h timeframe (previous 1d bar's levels available)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: 1.8x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA200(1d) and volume MA(20)
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend
        htf_1d_bullish = close[i] > ema_200_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_200_1d_aligned[i]
        
        if position == 0:
            # Look for Camarilla breakouts with volume confirmation
            long_breakout = (high[i] > camarilla_r1_aligned[i]) and volume_spike[i]
            short_breakout = (low[i] < camarilla_s1_aligned[i]) and volume_spike[i]
            
            # Only trade in direction of 1d trend
            if long_breakout and htf_1d_bullish:
                signals[i] = 0.25
                position = 1
            elif short_breakout and htf_1d_bearish:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price returns to Camarilla H3/L3 level or trend reverses
            camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 6.0
            camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
            exit_signal = (low[i] < camarilla_h3_aligned[i]) or (not htf_1d_bullish)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price returns to Camarilla L3/H3 level or trend reverses
            camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 6.0
            camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
            exit_signal = (high[i] > camarilla_l3_aligned[i]) or htf_1d_bullish
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeFilter_v2"
timeframe = "4h"
leverage = 1.0