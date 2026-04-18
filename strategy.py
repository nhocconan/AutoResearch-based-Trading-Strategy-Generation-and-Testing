#!/usr/bin/env python3
"""
1h_Camarilla_Pivot_R1_S1_Breakout_Volume_4hTrend
Hypothesis: Camarilla pivot R1/S1 breakout on 1h with volume confirmation and 4h trend filter.
Trade breakouts of R1 (resistance 1) and S1 (support 1) from previous day's Camarilla pivot levels.
Only take long when price breaks above R1 with volume > 1.5x 24-period average and 4h close > 4h open (bullish).
Only take short when price breaks below S1 with volume > 1.5x 24-period average and 4h close < 4h open (bearish).
Session filter: 08-20 UTC to avoid low-volume periods. Fixed position size 0.20.
Designed for 15-30 trades/year (~60-120 over 4 years) to avoid fee drag. Works in bull/bear by following intraday breakouts with trend alignment.
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
    
    # Get daily data for Camarilla pivot calculation (previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_r1 = np.full_like(close_1d, np.nan)
    camarilla_s1 = np.full_like(close_1d, np.nan)
    
    valid = ~(np.isnan(high_1d) | np.isnan(low_1d) | np.isnan(close_1d))
    camarilla_r1[valid] = close_1d[valid] + 1.1 * (high_1d[valid] - low_1d[valid]) / 12
    camarilla_s1[valid] = close_1d[valid] - 1.1 * (high_1d[valid] - low_1d[valid]) / 12
    
    # Align Camarilla levels to 1h timeframe (previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    
    # 4h trend: bullish if close > open, bearish if close < open
    bullish_4h = close_4h > open_4h
    bearish_4h = close_4h < open_4h
    
    # Align 4h trend to 1h timeframe
    bullish_4h_aligned = align_htf_to_ltf(prices, df_4h, bullish_4h.astype(float))
    bearish_4h_aligned = align_htf_to_ltf(prices, df_4h, bearish_4h.astype(float))
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    volume_confirm = volume > 1.5 * vol_ma
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    start_idx = max(24, 1)  # Need volume MA and at least 1 bar
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(bullish_4h_aligned[i]) or np.isnan(bearish_4h_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Check session
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Long: price breaks above R1 with volume and 4h bullish trend
        if close[i] > r1_aligned[i] and volume_confirm[i] and bullish_4h_aligned[i] > 0.5:
            signals[i] = 0.20
        # Short: price breaks below S1 with volume and 4h bearish trend
        elif close[i] < s1_aligned[i] and volume_confirm[i] and bearish_4h_aligned[i] > 0.5:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Camarilla_Pivot_R1_S1_Breakout_Volume_4hTrend"
timeframe = "1h"
leverage = 1.0