#!/usr/bin/env python3
name = "6h_ConnorsRSI_Pivot_Trend_Filter"
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
    
    # Load 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1d OHLC for daily pivots (previous day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (P, R1, S1)
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = pivot + (high_1d - low_1d)
    s1 = pivot - (high_1d - low_1d)
    
    # Align daily pivots to 6h (wait for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate Connors RSI (6h timeframe)
    # RSI(3)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    avg_loss = loss.ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    rs = avg_gain / avg_loss
    rsi3 = 100 - (100 / (1 + rs))
    
    # RSI of streak length (2-period)
    up_days = np.where(delta > 0, 1, 0)
    down_days = np.where(delta < 0, 1, 0)
    streak_up = np.where(up_days, np.maximum.accumulate(up_days * (np.arange(len(up_days)) + 1)), 0)
    streak_down = np.where(down_days, np.maximum.accumulate(down_days * (np.arange(len(down_days)) + 1)), 0)
    streak = streak_up - streak_down
    streak_rsi = pd.Series(streak).rolling(window=2, min_periods=2).apply(lambda x: 100 if x[0] < x[1] else 0, raw=False)
    streak_rsi = streak_rsi.fillna(50).values  # neutral when no streak
    
    # Percent Rank of RSI(3) over 100 periods
    rsi3_series = pd.Series(rsi3)
    percent_rank = rsi3_series.rolling(window=100, min_periods=100).apply(
        lambda x: (np.sum(x < x[-1]) / len(x)) * 100, raw=False
    ).fillna(50).values
    
    # Connors RSI = (RSI(3) + RSI(Streak) + PercentRank) / 3
    crsi = (rsi3 + streak_rsi + percent_rank) / 3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(crsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: CRSI < 15 (oversold) + price > 12h EMA50 (uptrend) + price > S1 (support)
            if (crsi[i] < 15 and close[i] > ema_50_12h_aligned[i] and close[i] > s1_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: CRSI > 85 (overbought) + price < 12h EMA50 (downtrend) + price < R1 (resistance)
            elif (crsi[i] > 85 and close[i] < ema_50_12h_aligned[i] and close[i] < r1_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: CRSI > 70 (overbought threshold) or price breaks below S1
            if (crsi[i] > 70) or (close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: CRSI < 30 (oversold threshold) or price breaks above R1
            if (crsi[i] < 30) or (close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals