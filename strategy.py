#!/usr/bin/env python3
name = "12h_1w_1d_Camarilla_R1S1_Breakout_Trend_Volume"
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
    
    # Weekly trend: EMA(8) on weekly close
    ema_8_1w = pd.Series(df_1w['close']).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_8_1w)
    
    # Daily trend: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily Camarilla pivot levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    s1 = prev_close - (range_hl * 1.08 / 2)
    r1 = prev_close + (range_hl * 1.08 / 2)
    
    # Align daily levels to 12h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Volume spike detection: 4-period average (2 days of 12h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 4)  # Wait for daily EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_8_1w_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and weekly/daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            weekly_uptrend = ema_8_1w_aligned[i] > ema_8_1w_aligned[i-1]
            daily_uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > s1_aligned[i] and vol_condition and weekly_uptrend and daily_uptrend:
                signals[i] = 0.30
                position = 1
            # Short: price below R1 with volume and weekly/daily downtrend
            elif close[i] < r1_aligned[i] and vol_condition and not weekly_uptrend and not daily_uptrend:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops
            if close[i] < s1_aligned[i] or volume[i] < vol_ma_4[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price back above R1 or volume drops
            if close[i] > r1_aligned[i] or volume[i] < vol_ma_4[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 12h Camarilla S1/R1 breakout with weekly/daily trend and volume confirmation
# - Weekly trend filter (EMA8) ensures alignment with major trend direction
# - Daily Camarilla S1/R1 act as strong support/resistance levels
# - Breakout above S1 with volume in weekly/daily uptrend = long opportunity
# - Breakdown below R1 with volume in weekly/daily downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to S1/R1 or volume weakens (1.5x average)
# - Position size 0.30 targets ~20-40 trades/year, avoiding fee drag
# - Uses multiple timeframe confirmation (weekly trend + daily levels) for robustness
# - Designed to work in BOTH bull and bear markets via trend filters
# - Tight volume conditions (2.0 entry, 1.5 exit) to reduce trade frequency and avoid overtrading