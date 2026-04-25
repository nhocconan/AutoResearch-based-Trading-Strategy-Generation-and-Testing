#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_1wRegime_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 12h with 1d EMA50 trend filter and 1-week choppiness regime. In bullish 1d trend and choppy weekly regime (range-bound), buy R1 breakouts; in bearish 1d trend and choppy weekly regime, sell S1 breakouts. Volume confirmation (1.8x 24-bar avg) filters low-quality breakouts. Designed for 12h timeframe with tight entries (~12-25/year) to minimize fee drag while capturing mean-reversion in weekly ranging markets with daily trend alignment.
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
    
    # Get 1d data for HTF trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for choppiness regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1-week choppiness index (CHOP)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14) and CHOP(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_denominator = highest_high_14 - lowest_low_14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop = 100 * np.log10(atr_14 * np.sqrt(14) / chop_denominator) / np.log10(14)
    
    # Align 1d EMA50 and 1w CHOP to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Calculate Camarilla levels using previous 12h bar's OHLC
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_r1 = close_12h + 1.1 * (high_12h - low_12h) / 12.0
    camarilla_s1 = close_12h - 1.1 * (high_12h - low_12h) / 12.0
    
    # Align Camarilla levels to 12h timeframe (previous 12h bar's levels available)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # Volume confirmation: 1.8x 24-bar average volume
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50(1d), CHOP(1w), and volume MA(24)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine regime and trend
        htfbullish = close[i] > ema_50_1d_aligned[i]
        htbearish = close[i] < ema_50_1d_aligned[i]
        choppy_regime = chop_aligned[i] > 50.0  # Chop > 50 indicates ranging market
        
        if position == 0:
            # Look for Camarilla breakouts with volume confirmation in choppy regime
            long_breakout = (high[i] > camarilla_r1_aligned[i]) and volume_spike[i]
            short_breakout = (low[i] < camarilla_s1_aligned[i]) and volume_spike[i]
            
            # Only trade in choppy regime with trend alignment
            if long_breakout and htfbullish and choppy_regime:
                signals[i] = 0.25
                position = 1
            elif short_breakout and htbearish and choppy_regime:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price returns to Camarilla H3 level or regime/trend changes
            camarilla_h3 = close_12h + 1.1 * (high_12h - low_12h) / 6.0
            camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
            exit_signal = (low[i] < camarilla_h3_aligned[i]) or (not htfbullish) or (not choppy_regime)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price returns to Camarilla L3 level or regime/trend changes
            camarilla_l3 = close_12h - 1.1 * (high_12h - low_12h) / 6.0
            camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
            exit_signal = (high[i] > camarilla_l3_aligned[i]) or htfbullish or (not choppy_regime)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_1wRegime_v1"
timeframe = "12h"
leverage = 1.0