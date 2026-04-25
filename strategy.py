#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeRegime
Hypothesis: Camarilla R1/S1 breakout on 12h with 1d EMA trend filter and volume confirmation.
Works in bull/bear markets by only trading breakouts aligned with daily trend.
Volume confirmation reduces false breakouts. Low trade frequency (~15-25/year) minimizes fee drag.
Designed for 12h timeframe target: 50-150 total trades over 4 years.
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
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate previous day's Camarilla levels (R1, S1)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    r1_1d = close_1d + camarilla_range
    s1_1d = close_1d - camarilla_range
    
    # Align HTF indicators to 12h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d, additional_delay_bars=1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d, additional_delay_bars=1)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if HTF data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals with trend and volume filters
            # Long: price breaks above R1 in uptrend (close > EMA50) with volume confirmation
            # Short: price breaks below S1 in downtrend (close < EMA50) with volume confirmation
            long_signal = (close[i] > r1_aligned[i]) and (close[i] > ema50_aligned[i]) and volume_confirm[i]
            short_signal = (close[i] < s1_aligned[i]) and (close[i] < ema50_aligned[i]) and volume_confirm[i]
            
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
            # Exit when price moves back below EMA50 (trend reversal) or breaks below S1 (mean reversion)
            exit_signal = (close[i] < ema50_aligned[i]) or (close[i] < s1_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above EMA50 (trend reversal) or breaks above R1 (mean reversion)
            exit_signal = (close[i] > ema50_aligned[i]) or (close[i] > r1_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeRegime"
timeframe = "12h"
leverage = 1.0