#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R1, S1) from previous day's range
    r1 = np.zeros_like(close_1d)
    s1 = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        r = ph - pl
        r1[i] = pc + (r * 1.0833)  # R1
        s1[i] = pc - (r * 1.0833)  # S1
    r1[0] = s1[0] = np.nan
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_trend_up = ema_21_1w[1:] > ema_21_1w[:-1]
    weekly_trend_up = np.concatenate([[False], weekly_trend_up])
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    
    # Volume confirmation: current volume > 2.0x 50-period EMA
    vol_ema = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for volume EMA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(weekly_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R1 in weekly uptrend with volume
            if (weekly_trend_aligned[i] > 0.5 and
                close[i] > r1_aligned[i] * 1.002 and  # 0.2% breakout filter
                vol_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short: break below S1 in weekly downtrend with volume
            elif (weekly_trend_aligned[i] <= 0.5 and
                  close[i] < s1_aligned[i] * 0.998 and  # 0.2% breakdown filter
                  vol_confirm[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: break below S1 or weekly trend turns down
            if close[i] < s1_aligned[i] * 1.002 or weekly_trend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: break above R1 or weekly trend turns up
            if close[i] > r1_aligned[i] * 0.998 or weekly_trend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals