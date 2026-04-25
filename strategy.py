#!/usr/bin/env python3
"""
1d_Camarilla_H3L3_Breakout_1wTrend_VolumeConfirm
Hypothesis: Daily Camarilla H3/L3 breakouts with weekly trend filter (EMA34) and volume spike (>1.5x 20-bar avg).
Only takes longs in weekly uptrend (close > EMA34) and shorts in weekly downtrend (close < EMA34).
Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (<25/year)
to work in both bull and bear markets via weekly trend alignment and volume confirmation.
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
    
    # Get 1d data for Camarilla levels (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1w for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w, additional_delay_bars=1)
    
    # Calculate Camarilla levels on 1d data (based on previous bar's OHLC)
    camarilla_h3_1d = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_l3_1d = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    camarilla_h4_1d = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    camarilla_l4_1d = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align HTF indicators to 1d timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d, additional_delay_bars=1)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d, additional_delay_bars=1)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d, additional_delay_bars=1)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d, additional_delay_bars=1)
    
    # Volume confirmation: 1.5x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals with trend filter and volume spike
            # Long: price breaks above H3 in uptrend (close > EMA34) with volume spike
            # Short: price breaks below L3 in downtrend (close < EMA34) with volume spike
            long_signal = (close[i] > camarilla_h3_aligned[i]) and (close[i] > ema34_1w_aligned[i]) and volume_spike[i]
            short_signal = (close[i] < camarilla_l3_aligned[i]) and (close[i] < ema34_1w_aligned[i]) and volume_spike[i]
            
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
            # Exit when price moves back below Camarilla L4 (stop loss at support)
            exit_signal = close[i] < camarilla_l4_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla H4 (stop loss at resistance)
            exit_signal = close[i] > camarilla_h4_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0