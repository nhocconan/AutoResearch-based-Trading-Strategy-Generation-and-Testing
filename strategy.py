#!/usr/bin/env python3
"""
1d_Camarilla_H4L4_Breakout_1wTrend_VolumeConfirm
Hypothesis: Daily Camarilla H4/L4 breakouts with weekly EMA34 trend filter and volume spike confirmation.
Uses 1-week EMA34 for trend direction to capture both bull and bear markets. Volume spike (>1.8x 20-bar average)
confirms breakout strength. Exits on reversion to Camarilla H3/L3 levels. Designed for low trade frequency
(7-25/year) to minimize fee drag and work in both bull and bear regimes.
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
    
    # Get weekly data for HTF trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate EMA34 on weekly close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels on weekly data (based on previous bar's OHLC)
    # H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    camarilla_h4_1w = close_1w + ((high_1w - low_1w) * 1.1 / 2)
    camarilla_l4_1w = close_1w - ((high_1w - low_1w) * 1.1 / 2)
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_h3_1w = close_1w + ((high_1w - low_1w) * 1.1 / 4)
    camarilla_l3_1w = close_1w - ((high_1w - low_1w) * 1.1 / 4)
    
    # Align HTF indicators to daily timeframe (completed weekly bar lag)
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w, additional_delay_bars=1)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4_1w, additional_delay_bars=1)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4_1w, additional_delay_bars=1)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w, additional_delay_bars=1)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w, additional_delay_bars=1)
    
    # Volume confirmation: 1.8x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals in direction of weekly trend with volume confirmation
            # Long: price breaks above H4 in uptrend (close > EMA34)
            # Short: price breaks below L4 in downtrend (close < EMA34)
            long_signal = (close[i] > camarilla_h4_aligned[i]) and (close[i] > ema34_aligned[i]) and volume_spike[i]
            short_signal = (close[i] < camarilla_l4_aligned[i]) and (close[i] < ema34_aligned[i]) and volume_spike[i]
            
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
            # Exit when price moves back below H3 (mean reversion to higher level)
            exit_signal = close[i] < camarilla_h3_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above L3 (mean reversion to lower level)
            exit_signal = close[i] > camarilla_l3_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_H4L4_Breakout_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0