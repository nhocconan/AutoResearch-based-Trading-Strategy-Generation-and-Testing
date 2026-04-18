#!/usr/bin/env python3
"""
12h_Pivot_R1_S1_Breakout_1wEMA34_Volume
Hypothesis: Uses Camarilla pivot levels (R1, S1) from daily data for breakout signals, with weekly EMA34 trend filter and volume confirmation.
Designed for low trade frequency (~15-25/year) with strong trend-following capability in both bull and bear markets.
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
    
    # Get daily data for pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R1, S1) from daily OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    camarilla_R1 = daily_close + (daily_high - daily_low) * 1.1 / 12
    camarilla_S1 = daily_close - (daily_high - daily_low) * 1.1 / 12
    
    # Align pivot levels to 12h timeframe (wait for daily close)
    R1_12h = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    S1_12h = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Weekly EMA34 for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    # Calculate EMA(34) on weekly close
    ema34_weekly = np.full(len(weekly_close), np.nan)
    k = 2 / (34 + 1)
    for i in range(len(weekly_close)):
        if i == 0:
            ema34_weekly[i] = weekly_close[i]
        elif np.isnan(ema34_weekly[i-1]):
            ema34_weekly[i] = weekly_close[i]
        else:
            ema34_weekly[i] = weekly_close[i] * k + ema34_weekly[i-1] * (1 - k)
    
    # Align weekly EMA34 to 12h timeframe
    ema34_12h = align_htf_to_ltf(prices, df_1w, ema34_weekly)
    
    # Volume confirmation: current volume > 1.8x 24-period average (2 days)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or np.isnan(ema34_12h[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with weekly uptrend and volume spike
            if close[i] > R1_12h[i] and close[i-1] <= R1_12h[i-1] and ema34_12h[i] > ema34_12h[i-1] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with weekly downtrend and volume spike
            elif close[i] < S1_12h[i] and close[i-1] >= S1_12h[i-1] and ema34_12h[i] < ema34_12h[i-1] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price breaks below S1 or weekly trend turns down
            if close[i] < S1_12h[i] or ema34_12h[i] <= ema34_12h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price breaks above R1 or weekly trend turns up
            if close[i] > R1_12h[i] or ema34_12h[i] >= ema34_12h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_1wEMA34_Volume"
timeframe = "12h"
leverage = 1.0