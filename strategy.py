#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrend_VolumeConfirm
Hypothesis: Camarilla R1/S1 breakouts on 1d with 1w EMA50 trend filter and volume confirmation.
R1/S1 represent the inner Camarilla levels - breakouts here with trend and volume have high continuation probability.
Uses 1w EMA50 for trend direction (stable weekly trend) to filter breakouts.
Volume spike (>1.5x 20-bar average) confirms participation. Exits on reversion to Camarilla H4/L4 levels.
Discrete position sizing (0.25) minimizes fee churn. Target: 15-30 trades/year.
Works in bull markets (breakouts with trend) and bear markets (breakouts against trend via short signals).
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for Camarilla levels (based on previous bar's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels on 1d data
    # R1 = Close + (High-Low)*1.1/12
    # S1 = Close - (High-Low)*1.1/12
    # H4 = Close + (High-Low)*1.1/2
    # L4 = Close - (High-Low)*1.1/2
    camarilla_r1_1d = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    camarilla_s1_1d = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    camarilla_h4_1d = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    camarilla_l4_1d = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align HTF indicators to 1d timeframe (completed 1w bar lag)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Align 1d indicators to 1d timeframe (completed 1d bar lag)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d, additional_delay_bars=1)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d, additional_delay_bars=1)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d, additional_delay_bars=1)
    
    # Volume confirmation: 1.5x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50
    start_idx = 55
    
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
            # Look for breakout signals in direction of 1w trend with volume confirmation
            # Long: price breaks above R1 in uptrend (close > EMA50)
            # Short: price breaks below S1 in downtrend (close < EMA50)
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

name = "1d_Camarilla_R1S1_Breakout_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0