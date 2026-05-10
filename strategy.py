#!/usr/bin/env python3
# 12h_KAMA_Donchian_Breakout_Volume
# Hypothesis: 12h KAMA trend filter with Donchian breakout and volume confirmation.
# Long when price breaks above Donchian upper (20) + price > KAMA + volume spike.
# Short when price breaks below Donchian lower (20) + price < KAMA + volume spike.
# Uses 1d trend for multi-timeframe alignment and volume confirmation to reduce false signals.
# Designed for low trade frequency (target: 12-37/year) with adaptive trend strength.

name = "12h_KAMA_Donchian_Breakout_Volume"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate KAMA on 12h close (adaptive moving average)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility > 0, change / volatility, 0)
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate daily EMA for trend filter (34-period)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10), EMA (34), Donchian (20), volume (20)
    start_idx = max(10, 34, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Donchian breakout
        breakout_up = close[i] > donchian_high[i-1]  # break above prior period high
        breakout_down = close[i] < donchian_low[i-1]  # break below prior period low
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        if position == 0:
            # Long entry: Donchian breakout up + price > KAMA + uptrend + volume spike
            if breakout_up and price_above_kama and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Donchian breakout down + price < KAMA + downtrend + volume spike
            elif breakout_down and price_below_kama and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA or Donchian breakdown
            if close[i] < kama[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA or Donchian breakout up
            if close[i] > kama[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals