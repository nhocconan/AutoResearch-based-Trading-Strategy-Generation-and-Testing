#!/usr/bin/env python3
name = "1h_4h_1d_Camarilla_S1R1_Breakout_Trend"
timeframe = "1h"
leverage = 1.0

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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    s1 = prev_close - (range_hl * 1.08 / 2)
    r1 = prev_close + (range_hl * 1.08 / 2)
    
    # Align daily levels to 1h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1h session filter (08-20 UTC)
    hours = prices.index.hour
    
    # Volume spike detection: 6-period average (6 hours)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 6)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 8-20
        
        if position == 0:
            # Long: price above S1 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_6[i] * 1.8
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > s1_aligned[i] and vol_condition and uptrend and in_session:
                signals[i] = 0.20
                position = 1
            # Short: price below R1 with volume and daily downtrend
            elif close[i] < r1_aligned[i] and vol_condition and not uptrend and in_session:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops or outside session
            if close[i] < s1_aligned[i] or volume[i] < vol_ma_6[i] * 1.2 or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price back above R1 or volume drops or outside session
            if close[i] > r1_aligned[i] or volume[i] < vol_ma_6[i] * 1.2 or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla S1/R1 breakout with daily trend and volume confirmation
# - Daily Camarilla S1/R1 act as strong support/resistance levels
# - Breakout above S1 with volume in daily uptrend = long opportunity
# - Breakdown below R1 with volume in daily downtrend = short opportunity
# - Volume spike (1.8x average) confirms institutional participation
# - Session filter (08-20 UTC) reduces noise trades
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to S1/R1 or volume weakens or outside session
# - Position size 0.20 targets ~15-30 trades/year, avoiding fee drag
# - Uses actual daily Camarilla levels (not weekly) for better responsiveness
# - Designed to work in BOTH bull and bear markets via trend filter
# - Uses 1h timeframe for precise entry timing with 4h/1d for signal direction