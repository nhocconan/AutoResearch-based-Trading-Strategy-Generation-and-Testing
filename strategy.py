#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Breakout_1wTrend_VolumeConfirmation
Hypothesis: On daily timeframe, use weekly Camarilla pivot points (R1/S1) for breakout entries with weekly EMA34 trend filter and volume confirmation (>1.5x 20-period average). Daily provides lower trade frequency to minimize fee drag while weekly trend ensures alignment with higher timeframe momentum. Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need at least 34 periods for EMA34
        return np.zeros(n)
    
    # Calculate weekly OHLC for Camarilla pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels (based on previous weekly bar's range)
    # Camarilla R1 = close + 1.1*(high - low)/12
    # Camarilla S1 = close - 1.1*(high - low)/12
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    
    # Set first value to NaN (no previous bar)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    camarilla_r1 = prev_close_1w + 1.1 * (prev_high_1w - prev_low_1w) / 12
    camarilla_s1 = prev_close_1w - 1.1 * (prev_high_1w - prev_low_1w) / 12
    
    # Calculate weekly EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA to daily timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA34 + volume MA
    start_idx = max(34, 20)  # 34 for EMA34, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend alignment
        trend_1w_uptrend = close[i] > ema_34_1w_aligned[i]
        trend_1w_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + weekly uptrend + volume confirmation
            long_signal = (close[i] > camarilla_r1_aligned[i]) and trend_1w_uptrend and volume_spike[i]
            
            # Short: price breaks below S1 + weekly downtrend + volume confirmation
            short_signal = (close[i] < camarilla_s1_aligned[i]) and trend_1w_downtrend and volume_spike[i]
            
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
            # Exit: price touches S1 OR weekly trend turns down
            if (close[i] < camarilla_s1_aligned[i] or not trend_1w_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches R1 OR weekly trend turns up
            if (close[i] > camarilla_r1_aligned[i] or not trend_1w_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_Pivot_Breakout_1wTrend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0