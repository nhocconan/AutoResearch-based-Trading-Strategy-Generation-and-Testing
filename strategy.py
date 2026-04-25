#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirm
Hypothesis: Camarilla R1/S1 breakouts on 12h with 1d EMA50 trend filter and volume confirmation.
R1/S1 represent the inner Camarilla levels - breakouts here with trend and volume have high continuation probability.
Uses 1d EMA50 for trend direction (stable daily trend) to filter breakouts.
Volume spike (>1.5x 20-bar average) confirms participation. Exits on reversion to Camarilla H4/L4 levels.
Discrete position sizing (0.25) minimizes fee churn. Target: 12-37 trades/year.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe (completed 1d bar lag)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    
    # Get 12h data for Camarilla levels (based on previous bar's OHLC)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Camarilla levels on 12h data
    # R1 = Close + (High-Low)*1.1/12
    # S1 = Close - (High-Low)*1.1/12
    # H4 = Close + (High-Low)*1.1/2
    # L4 = Close - (High-Low)*1.1/2
    camarilla_r1_12h = close_12h + ((high_12h - low_12h) * 1.1 / 12)
    camarilla_s1_12h = close_12h - ((high_12h - low_12h) * 1.1 / 12)
    camarilla_h4_12h = close_12h + ((high_12h - low_12h) * 1.1 / 2)
    camarilla_l4_12h = close_12h - ((high_12h - low_12h) * 1.1 / 2)
    
    # Align 12h indicators to 12h timeframe (completed 12h bar lag)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1_12h, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1_12h, additional_delay_bars=1)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4_12h, additional_delay_bars=1)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4_12h, additional_delay_bars=1)
    
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
            # Look for breakout signals in direction of 1d trend with volume confirmation
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

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0