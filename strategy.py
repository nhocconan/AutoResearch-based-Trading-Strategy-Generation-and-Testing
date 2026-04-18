#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_Volume
Hypothesis: Trade Donchian channel breakouts on 12h with 1d trend confirmation and volume filter.
In bull markets: buy breakouts above 20-period high when 1d EMA50 > EMA200 (uptrend).
In bear markets: sell breakdowns below 20-period low when 1d EMA50 < EMA200 (downtrend).
Volume must exceed 1.5x 24-period average to confirm breakout strength.
Uses discrete position sizing (0.25) to limit fee churn. Designed for 12-37 trades/year.
Works in both bull/bear by following 1d trend direction - avoids counter-trend trades.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 and EMA200 for trend filter
    ema50_1d = np.full_like(df_1d['close'].values, np.nan)
    ema200_1d = np.full_like(df_1d['close'].values, np.nan)
    
    close_1d = df_1d['close'].values
    if len(close_1d) >= 200:
        # EMA50
        alpha50 = 2 / (50 + 1)
        ema50_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema50_1d[i] = alpha50 * close_1d[i] + (1 - alpha50) * ema50_1d[i-1]
        
        # EMA200
        alpha200 = 2 / (200 + 1)
        ema200_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema200_1d[i] = alpha200 * close_1d[i] + (1 - alpha200) * ema200_1d[i-1]
    
    # Align 1d EMAs to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Donchian channel (20-period) on 12h
    lookback = 20
    donchian_high = np.full_like(close, np.nan)
    donchian_low = np.full_like(close, np.nan)
    
    if len(close) >= lookback:
        for i in range(lookback - 1, len(close)):
            donchian_high[i] = np.max(high[i - lookback + 1:i + 1])
            donchian_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, vol_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: 1 = uptrend (EMA50 > EMA200), -1 = downtrend (EMA50 < EMA200)
        trend = 0
        if ema50_1d_aligned[i] > ema200_1d_aligned[i]:
            trend = 1
        elif ema50_1d_aligned[i] < ema200_1d_aligned[i]:
            trend = -1
        
        if position == 0:
            # Long: breakout above Donchian high + uptrend + volume
            if close[i] > donchian_high[i] and trend == 1 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low + downtrend + volume
            elif close[i] < donchian_low[i] and trend == -1 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below Donchian low or trend turns down
            if close[i] < donchian_low[i] or trend == -1:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above Donchian high or trend turns up
            if close[i] > donchian_high[i] or trend == 1:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0