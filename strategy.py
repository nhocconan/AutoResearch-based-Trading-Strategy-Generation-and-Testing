#!/usr/bin/env python3
name = "6h_ConnorsRSI_MeanReversion_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rsi(series, period):
    """Calculate RSI with given period."""
    delta = np.diff(series, prepend=series[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Calculate Connors RSI (CRSI)."""
    # RSI component
    rsi_val = rsi(close, rsi_period)
    
    # Streak component: consecutive up/down days
    streak = np.zeros_like(close)
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    # RSI of streak (absolute values)
    streak_abs = np.abs(streak)
    streak_rsi = rsi(streak_abs, streak_period)
    
    # Percent Rank component: where current close ranks vs past N closes
    percent_rank = np.zeros_like(close)
    for i in range(len(close)):
        if i < rank_period:
            percent_rank[i] = 50  # neutral when insufficient history
        else:
            window = close[i-rank_period+1:i+1]
            percent_rank[i] = np.sum(window < close[i]) / rank_period * 100
    
    # CRSI = average of three components
    crsi = (rsi_val + streak_rsi + percent_rank) / 3
    return crsi

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend filter: close above/below 1d EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # Calculate Connors RSI on 6h data
    crsi = connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Volume filter: volume > 1.3x 20-period average (6f)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.3 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for CRSI rank component
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(crsi[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: CRSI < 15 (oversold) + uptrend + volume filter
            if crsi[i] < 15 and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: CRSI > 85 (overbought) + downtrend + volume filter
            elif crsi[i] > 85 and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: CRSI > 60 (mean reversion) or trend down
            if crsi[i] > 60 or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: CRSI < 40 (mean reversion) or trend up
            if crsi[i] < 40 or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals