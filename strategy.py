#!/usr/bin/env python3
name = "12h_1w_1d_Camarilla_S1R1_Breakout_Trend"
timeframe = "12h"
leverage = 1.0

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
    
    # Load weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 10 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly trend: EMA(21) on weekly close
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate daily data for previous day's close (needed for Camarilla)
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    range_hl = prev_high_1d - prev_low_1d
    
    # Camarilla levels S1 and R1
    s1 = prev_close_1d - (range_hl * 1.08 / 2)
    r1 = prev_close_1d + (range_hl * 1.08 / 2)
    
    # Align daily levels to 12h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Volume spike detection: 4-period average (2 days of 12h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 4)  # Wait for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            weekly_uptrend = ema_21_1w_aligned[i] > ema_21_1w_aligned[i-1]
            
            if close[i] > s1_aligned[i] and vol_condition and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below R1 with volume and weekly downtrend
            elif close[i] < r1_aligned[i] and vol_condition and not weekly_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops
            if close[i] < s1_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R1 or volume drops
            if close[i] > r1_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla S1/R1 breakout with weekly trend and volume confirmation
# - Weekly EMA(21) filter ensures we only trade in the direction of the higher timeframe trend
# - Daily Camarilla S1/R1 act as strong support/resistance levels for 12h timeframe
# - Breakout above S1 with volume in weekly uptrend = long opportunity
# - Breakdown below R1 with volume in weekly downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to S1/R1 or volume weakens
# - Position size 0.25 targets ~15-30 trades/year, avoiding fee drag
# - Uses weekly trend filter to avoid counter-trend trades in choppy markets
# - Designed for 12h timeframe to balance trade frequency and signal quality