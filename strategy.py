#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1_S1_Breakout_Volume_1dTrendFilter
Hypothesis: Trade Camarilla pivot breakouts on 12h with 1d trend filter and volume confirmation. The Camarilla pivot system identifies key intraday support/resistance levels (R1, S1) based on prior day's range. In trending markets, price often breaks these levels with momentum. We filter for 1d trend using EMA34 to avoid counter-trend trades. Volume confirmation ensures breakout validity. Targets 15-30 trades/year via strict entry conditions. Works in bull/bear by following the 1d trend direction. Uses 1d EMA34 as trend filter to avoid whipsaw in sideways markets.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    ema_period = 34
    close_1d = df_1d['close'].values
    ema_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= ema_period:
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1))) + (ema_1d[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align 1d EMA34 to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We'll calculate these for each 1d bar and align to 12h
    if len(df_1d) >= 2:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        camarilla_r1 = np.full_like(close_1d, np.nan)
        camarilla_s1 = np.full_like(close_1d, np.nan)
        
        for i in range(1, len(close_1d)):
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            range_val = prev_high - prev_low
            
            camarilla_r1[i] = prev_close + (range_val * 1.1 / 12)
            camarilla_s1[i] = prev_close - (range_val * 1.1 / 12)
        
        # Align Camarilla levels to 12h timeframe
        r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
        s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    else:
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, vol_period)  # Need enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 + 1d uptrend (price > EMA34) + volume
            if close[i] > r1_aligned[i] and close[i] > ema_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + 1d downtrend (price < EMA34) + volume
            elif close[i] < s1_aligned[i] and close[i] < ema_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or 1d trend turns down
            if close[i] < s1_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or 1d trend turns up
            if close[i] > r1_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_R1_S1_Breakout_Volume_1dTrendFilter"
timeframe = "12h"
leverage = 1.0