#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h Camarilla R1/S1 breakout with 1d trend filter (price > 1d EMA50 for long, < 1d EMA50 for short) and volume confirmation (>2.0x 24-bar mean volume). Uses HTF 1d for trend alignment to capture longer-term momentum while reducing whipsaw. Volume confirmation ensures breakouts have conviction. Discrete position sizing (0.25) minimizes fee churn. Designed for 12-25 trades/year per symbol, effective in both bull (breakouts with volume) and bear (trend-following via shorts) markets. 12h timeframe targets lower trade frequency to reduce fee drag while maintaining responsiveness.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA(50) on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of prior bar)
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d)  # R1 = C + 1.1*(H-L)
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d)  # S1 = C - 1.1*(H-L)
    
    # Align Camarilla levels to 12h timeframe (use previous bar's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 2.0x 24-bar mean volume (24*12h = 12d lookback)
    vol_mean_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > (vol_mean_24 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA and volume mean
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_mean_24[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 in uptrend (price > 1d EMA50) with volume confirmation
            # Short: price breaks below Camarilla S1 in downtrend (price < 1d EMA50) with volume confirmation
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema_50_aligned[i]) and vol_confirm[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema_50_aligned[i]) and vol_confirm[i]
            
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
            # Exit when price moves back below 1d EMA50 (trend reversal)
            exit_signal = close[i] < ema_50_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above 1d EMA50 (trend reversal)
            exit_signal = close[i] > ema_50_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0