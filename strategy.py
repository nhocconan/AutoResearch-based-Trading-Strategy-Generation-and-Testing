#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_HTFRegime_v3
Hypothesis: Trade Camarilla R1/S1 breakouts on 12h with 1w EMA50 trend filter and choppiness regime filter. In bullish 1w trend (price > EMA50) and low chop (CHOP < 50), buy breakouts above R1; in bearish 1w trend (price < EMA50) and low chop, sell breakdowns below S1. Uses volume spike (2.0x 24-bar avg) to confirm. Designed for 12h timeframe with very tight entries (~8-12/year) to minimize fee drag while capturing strong directional moves in both bull and bear markets. Uses discrete position sizing (0.25) and Camarilla H3/L3 for exits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla pivot calculation and choppiness filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12.0
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12.0
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 6.0  # H3 for long exit
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 6.0  # L3 for short exit
    
    # Align Camarilla levels to 12h timeframe (yesterday's levels available today)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate Choppiness Index on 1d (CHOP > 50 = choppy, CHOP < 50 = trending)
    # True Range = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # first bar
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_denominator = highest_high_14 - lowest_low_14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop = 100 * np.log10(atr_14 * np.sqrt(14) / chop_denominator) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: 2.0x 24-bar average volume (tighter than before)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for volume MA(24) and chop calculation
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend and chop regime
        htf_1w_bullish = close[i] > ema_50_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_50_1w_aligned[i]
        low_chop = chop_aligned[i] < 50.0  # Trending market (not choppy)
        
        if position == 0:
            # Look for Camarilla breakouts with volume confirmation and low chop
            long_breakout = (high[i] > camarilla_r1_aligned[i]) and volume_spike[i]
            short_breakout = (low[i] < camarilla_s1_aligned[i]) and volume_spike[i]
            
            # Only trade in direction of 1w trend AND low chop regime
            if long_breakout and htf_1w_bullish and low_chop:
                signals[i] = 0.25
                position = 1
            elif short_breakout and htf_1w_bearish and low_chop:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price returns to Camarilla H3 level or trend/chop deteriorates
            exit_signal = (low[i] < camarilla_h3_aligned[i]) or (not htf_1w_bullish) or (not low_chop)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price returns to Camarilla L3 level or trend/chop deteriorates
            exit_signal = (high[i] > camarilla_l3_aligned[i]) or htf_1w_bullish or (not low_chop)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_HTFRegime_v3"
timeframe = "12h"
leverage = 1.0