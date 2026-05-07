#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hEMA21_Trend_VolumeS_v1
Hypothesis: Trade breakouts of daily Camarilla R1/S1 levels only when aligned with 12h EMA21 trend and confirmed by volume spike. Uses discrete sizing and exits on trend reversal or price return to pivot to limit trades and reduce fee drag. Designed for 4h timeframe with 12h trend filter to work in both bull and bear markets.
"""

name = "4h_Camarilla_R1S1_Breakout_12hEMA21_Trend_VolumeS_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation (pivot based on prior day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot points (using prior day's OHLC)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    camarilla_range = daily_high - daily_low
    r1 = daily_close + (camarilla_range * 1.1 / 12)
    s1 = daily_close - (camarilla_range * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe (with 1-day delay for completed bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1, additional_delay_bars=1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1, additional_delay_bars=1)
    
    # Get 12h trend filter (EMA21)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Get 12h close for trend direction
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    # Get 4h volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_21_12h_aligned[i]) or 
            np.isnan(close_12h_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend
        trend_up = close_12h_aligned[i] > ema_21_12h_aligned[i]
        trend_down = close_12h_aligned[i] < ema_21_12h_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above R1 with upward trend and volume spike
            if (close[i] > r1_aligned[i] and 
                trend_up and 
                vol_ratio[i] > 3.0):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with downward trend and volume spike
            elif (close[i] < s1_aligned[i] and 
                  trend_down and 
                  vol_ratio[i] > 3.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to daily close or trend turns down
            if close[i] < daily_close[-1] if i == len(prices)-1 else False or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to daily close or trend turns up
            if close[i] > daily_close[-1] if i == len(prices)-1 else False or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals