#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Volume_And_1wTrend_Filter
Hypothesis: Combines KAMA trend direction with 1w EMA trend filter and volume confirmation.
Designed for low trade frequency (<10/year) to minimize fee burn while capturing
strong directional moves by requiring alignment across multiple timeframes and volume.
Should work in both bull and bear markets by filtering trades with higher timeframe trend.
"""

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
    
    # Get daily data for KAMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    # KAMA parameters: ER period=10, fast=2, slow=30
    change = np.abs(np.diff(df_1d['close'], prepend=df_1d['close'][0]))
    volatility = np.abs(np.diff(df_1d['close'])).rolling(window=10, min_periods=1).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(df_1d['close'])
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, len(df_1d)):
        kama[i] = kama[i-1] + sc[i] * (df_1d['close'].iloc[i] - kama[i-1])
    
    # Calculate 1w EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align KAMA and weekly EMA to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: current volume > 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filters
        kama_bullish = close[i] > kama_aligned[i]
        kama_bearish = close[i] < kama_aligned[i]
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_filter[i]
        
        # Entry conditions: KAMA crossover with weekly trend and volume
        long_entry = kama_bullish and weekly_uptrend and vol_confirm
        short_entry = kama_bearish and weekly_downtrend and vol_confirm
        
        # Exit conditions: opposite KAMA crossover
        long_exit = kama_bearish
        short_exit = kama_bullish
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_Trend_With_Volume_And_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0