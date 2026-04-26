#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike
Hypothesis: On 12h timeframe, enter long when price breaks above Camarilla R1 level AND 1w trend is up (close > EMA50) AND volume > 2x 20-period average. Enter short when price breaks below S1 level AND 1w trend is down (close < EMA50) AND volume spike. Uses Camarilla pivot levels from 1d for intraday structure, 1w EMA50 for higher timeframe trend alignment, and volume confirmation to filter noise. Designed for low trade frequency (12-37/year) to avoid fee drag while capturing strong trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1_1d = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 12
    camarilla_s1_1d = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 12
    
    # Align to 12h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d.values)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d.values)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup (50), volume MA warmup (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions relative to Camarilla levels
        breakout_above_r1 = close[i] > camarilla_r1_aligned[i]
        breakout_below_s1 = close[i] < camarilla_s1_aligned[i]
        
        # 1w trend filter
        trend_uptrend = close[i] > ema_50_1w_aligned[i]
        trend_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price above R1 + 1w uptrend + volume spike
            long_signal = breakout_above_r1 and trend_uptrend and volume_spike[i]
            
            # Short: price below S1 + 1w downtrend + volume spike
            short_signal = breakout_below_s1 and trend_downtrend and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below S1 OR trend change to downtrend
            if breakout_below_s1 or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R1 OR trend change to uptrend
            if breakout_above_r1 or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0