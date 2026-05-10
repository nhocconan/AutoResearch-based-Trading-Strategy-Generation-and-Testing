#!/usr/bin/env python3
"""
1d_KAMA_Trend_1dRSI_WeeklyVolumeFilter
Hypothesis: Use weekly volume filter to confirm institutional interest, KAMA for trend direction on daily timeframe, and RSI for entry timing. 
KAMA adapts to market noise, reducing whipsaws in ranging markets. Weekly volume filter ensures trades occur with participation. 
Targets 10-25 trades/year by requiring alignment of trend, momentum, and volume confirmation. Works in bull/bear via adaptive trend filter.
"""

name = "1d_KAMA_Trend_1dRSI_WeeklyVolumeFilter"
timeframe = "1d"
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
    
    # Get weekly data for volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly average volume for volume filter
    vol_avg_1w = pd.Series(df_1w['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_1w)
    
    # Calculate KAMA (adaptive trend) on daily close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle first 10 values where diff is not available
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(9, np.nan), volatility[9:]])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI (14) on daily close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first value
    rsi = np.concatenate([np.array([np.nan]), rsi])
    
    # Align weekly volume average to daily timeframe
    vol_avg_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10), RSI (14), volume avg (20)
    start_idx = max(10, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_avg_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below KAMA
        uptrend = close[i] > kama[i]
        downtrend = close[i] < kama[i]
        
        # Volume filter: current daily volume > 1.5x average weekly volume
        vol_daily = volume[i]
        vol_avg = vol_avg_1w_aligned[i]
        volume_filter = vol_daily > vol_avg * 1.5
        
        # RSI conditions for entry
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        if position == 0:
            # Long entry: uptrend + RSI oversold + volume filter
            if uptrend and rsi_oversold and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + RSI overbought + volume filter
            elif downtrend and rsi_overbought and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or RSI overbought
            if not uptrend or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or RSI oversold
            if not downtrend or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals