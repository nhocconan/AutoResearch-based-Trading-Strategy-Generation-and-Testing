#!/usr/bin/env python3
name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
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
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    trend_up = close > ema34_1d_aligned
    trend_down = close < ema34_1d_aligned
    
    # Daily Camarilla pivot levels (R1/S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].shift(1).values
    close_1d_prev = np.concatenate([[close_1d_prev[0]], close_1d_prev[:-1]])
    
    R1 = close_1d_prev + (high_1d - low_1d) * 1.1 / 12
    S1 = close_1d_prev - (high_1d - low_1d) * 1.1 / 12
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above R1 with volume surge and 1d uptrend
            if close[i] > R1_aligned[i] and vol_surge[i] and trend_up[i]:
                signals[i] = 0.30
                position = 1
            # Short: Close breaks below S1 with volume surge and 1d downtrend
            elif close[i] < S1_aligned[i] and vol_surge[i] and trend_down[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: Close below S1 or trend turns down
            if close[i] < S1_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: Close above R1 or trend turns up
            if close[i] > R1_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Camarilla R1/S1 breakouts with 1d trend filter and volume surge capture institutional breakout moves.
# Long when price breaks above R1 (first resistance) with volume confirmation in 1d uptrend.
# Short when price breaks below S1 (first support) with volume confirmation in 1d downtrend.
# Uses daily Camarilla levels for institutional relevance, 1d EMA34 for trend, and volume surge for conviction.
# Designed for 12h timeframe to balance trade frequency (~12-37/year) and capture multi-day trends.
# Works in bull markets (breaks above R1 in uptrend) and bear markets (breaks below S1 in downtrend).