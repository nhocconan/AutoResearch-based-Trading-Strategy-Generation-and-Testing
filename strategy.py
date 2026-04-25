#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrend_VolumeConfirm
Hypothesis: Daily Camarilla R1/S1 breakouts with 1-week EMA34 trend filter and volume spike confirmation.
Uses 1-week EMA34 for primary trend direction (works in both bull and bear markets).
Volume spike (>2x 20-bar average) confirms breakout strength. Exits on reversion to Camarilla C (midpoint).
Discrete position sizing (0.25) minimizes fee churn. Target: 7-25 trades/year (30-100 over 4 years).
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
    
    # Get 1w data for HTF trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate EMA34 on 1w close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels on 1w data (based on previous bar's OHLC)
    camarilla_r1_1w = close_1w + ((high_1w - low_1w) * 1.1 / 12)
    camarilla_s1_1w = close_1w - ((high_1w - low_1w) * 1.1 / 12)
    camarilla_c_1w = close_1w  # Camarilla C is the close
    
    # Align HTF indicators to 1d timeframe (completed 1w bar lag)
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w, additional_delay_bars=1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1_1w, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1_1w, additional_delay_bars=1)
    camarilla_c_aligned = align_htf_to_ltf(prices, df_1w, camarilla_c_1w, additional_delay_bars=1)
    
    # Volume confirmation: 2.0x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_c_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals in direction of 1w trend with volume confirmation
            # Long: price breaks above R1 in uptrend (close > EMA34)
            # Short: price breaks below S1 in downtrend (close < EMA34)
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema34_aligned[i]) and volume_spike[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema34_aligned[i]) and volume_spike[i]
            
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
            # Exit when price moves back below Camarilla C (mean reversion to midpoint)
            exit_signal = close[i] < camarilla_c_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla C (mean reversion to midpoint)
            exit_signal = close[i] > camarilla_c_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0