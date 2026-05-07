#!/usr/bin/env python3
name = "6h_1w_1d_RangeBreakout_VolumeTrend"
timeframe = "6h"
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
    
    # Load weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 10 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly range for trend context (using previous week's close)
    weekly_close = df_1w['close'].shift(1).values
    weekly_open = df_1w['open'].shift(1).values
    weekly_trend = weekly_close > weekly_open  # True if bullish week
    
    # Daily volatility filter: ATR(14) for breakout strength
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # Daily range for breakout levels (using previous day)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    range_hl = prev_high - prev_low
    
    # Breakout levels: previous day's high/low + 0.5 * ATR
    breakout_up = prev_high + 0.5 * atr_14
    breakout_down = prev_low - 0.5 * atr_14
    
    # Align weekly trend and breakout levels to 6h timeframe
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    breakout_up_aligned = align_htf_to_ltf(prices, df_1d, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_1d, breakout_down)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 4)  # Wait for ATR and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_trend_aligned[i]) or np.isnan(breakout_up_aligned[i]) or 
            np.isnan(breakout_down_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above previous day's high + 0.5*ATR with volume in bullish weekly trend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            if close[i] > breakout_up_aligned[i] and vol_condition and weekly_trend_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below previous day's low - 0.5*ATR with volume in bearish weekly trend
            elif close[i] < breakout_down_aligned[i] and vol_condition and not weekly_trend_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to breakout level or volume drops
            if close[i] < breakout_up_aligned[i] or volume[i] < vol_ma_4[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to breakout level or volume drops
            if close[i] > breakout_down_aligned[i] or volume[i] < vol_ma_4[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h range breakout with weekly trend filter and volume confirmation
# - Uses previous day's high/low + 0.5*ATR(14) as dynamic breakout levels
# - Weekly trend filter (bullish/bearish based on weekly open/close) ensures directional bias
# - Volume spike (2.0x 4-bar average) confirms institutional participation
# - Works in both bull and bear markets by aligning with weekly trend direction
# - Exit when price returns to breakout level or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Novelty: Combines intraday breakout with weekly trend context (not just daily)
# - Avoids saturated families by using ATR-scaled breakouts instead of fixed pivot levels