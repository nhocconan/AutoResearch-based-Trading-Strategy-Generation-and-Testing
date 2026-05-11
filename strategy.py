#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
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
    
    # 12h trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    trend_12h_up = close > ema_50_12h_aligned
    
    # Volume confirmation (20-period SMA)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * volume_ma20
    
    # Daily Camarilla levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    R1 = close_1d + (high_1d - low_1d) * 1.0833 / 12
    S1 = close_1d - (high_1d - low_1d) * 1.0833 / 12
    
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Breakout signals
    long_breakout = close > R1_aligned
    short_breakout = close < S1_aligned
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(trend_12h_up[i]) or np.isnan(volume_filter[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 12h uptrend + volume + break above R1
            if trend_12h_up[i] and volume_filter[i] and long_breakout[i]:
                signals[i] = 0.25
                position = 1
            # Short: 12h downtrend + volume + break below S1
            elif not trend_12h_up[i] and volume_filter[i] and short_breakout[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend reversal or volume drop
            if not trend_12h_up[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend reversal or volume drop
            if trend_12h_up[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals