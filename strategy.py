#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 breakout on 4h with 12h EMA50 trend filter and volume spike confirmation. 
Only trade breakouts in direction of 12h trend with volume > 1.5x 20-period average. 
Designed for low trade frequency (~20-35/year) to work in both bull and bear markets via trend alignment.
Uses discrete position sizing (0.25) to minimize fee churn. Camarilla levels provide institutional 
support/resistance with higher reliability than Donchian in ranging markets, while trend filter 
avoids counter-trend trades. Volume spike confirms institutional participation.
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
    
    # Get 12h data for HTF trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h, additional_delay_bars=1)
    
    # Get 1d data for Camarilla pivot levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar: based on previous day's OHLC
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    camarilla_R1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_S1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align to 4h timeframe (standard 1-bar delay for daily levels)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1, additional_delay_bars=1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1, additional_delay_bars=1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or
            np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for Camarilla breakout signals with trend and volume filters
            # Long: price breaks above R1 in uptrend (close > EMA50_12h) with volume spike
            # Short: price breaks below S1 in downtrend (close < EMA50_12h) with volume spike
            long_signal = (close[i] > camarilla_R1_aligned[i]) and \
                          (close[i] > ema50_12h_aligned[i]) and \
                          volume_spike[i]
            short_signal = (close[i] < camarilla_S1_aligned[i]) and \
                           (close[i] < ema50_12h_aligned[i]) and \
                           volume_spike[i]
            
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
            # Exit when price moves back below R1 (failed breakout) or trend reverses
            exit_signal = (close[i] < camarilla_R1_aligned[i]) or (close[i] < ema50_12h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above S1 (failed breakout) or trend reverses
            exit_signal = (close[i] > camarilla_S1_aligned[i]) or (close[i] > ema50_12h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0