#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrend_VolumeConfirm
Hypothesis: Camarilla R1/S1 breakouts on 12h with 1-week EMA50 trend filter and volume spike confirmation.
Uses 1-week EMA50 for primary trend direction (captures multi-week trends) and 12h Camarilla levels for precise entries.
Volume spike (>2x 20-bar average) confirms breakout strength. Exits on reversion to Camarilla C (midpoint).
Discrete position sizing (0.25) minimizes fee churn. Target: 12-30 trades/year.
Works in both bull and bear markets by following the 1-week trend.
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Camarilla levels on 12h data (based on previous bar's OHLC)
    camarilla_r1_12h = close_12h + ((high_12h - low_12h) * 1.1 / 2) * (1/4)  # R1 = Close + (H-L)*1.1/4
    camarilla_s1_12h = close_12h - ((high_12h - low_12h) * 1.1 / 2) * (1/4)  # S1 = Close - (H-L)*1.1/4
    camarilla_c_12h = close_12h  # Camarilla C is the close
    
    # Align HTF indicators to 12h timeframe (completed 1w bar lag)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Align 12h indicators to 12h timeframe (completed 12h bar lag - no additional delay needed for price-based levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1_12h, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1_12h, additional_delay_bars=1)
    camarilla_c_aligned = align_htf_to_ltf(prices, df_12h, camarilla_c_12h, additional_delay_bars=1)
    
    # Volume confirmation: 2.0x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_c_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals in direction of 1w trend
            # Long: price breaks above R1 in uptrend (close > EMA50_1w)
            # Short: price breaks below S1 in downtrend (close < EMA50_1w)
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema50_1w_aligned[i]) and volume_spike[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema50_1w_aligned[i]) and volume_spike[i]
            
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

name = "12h_Camarilla_R1S1_Breakout_1wTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0