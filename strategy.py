#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_Volume_TrendFilter
Hypothesis: Donchian(20) breakout on 12h with 1d trend filter and volume confirmation.
Long when price breaks above 20-period high with volume > 1.5x average and 1d close > 1d EMA50.
Short when price breaks below 20-period low with volume > 1.5x average and 1d close < 1d EMA50.
Designed for 12h timeframe to target 15-35 trades/year with tight entry conditions.
Works in bull markets by capturing breakouts and in bear markets by capturing breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    ema = np.zeros_like(close)
    if len(close) >= period:
        ema[period-1] = np.mean(close[:period])
        for i in range(period, len(close)):
            ema[i] = (close[i] * 2 / (period + 1)) + (ema[i-1] * (period - 1) / (period + 1))
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = calculate_ema(close_1d, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Donchian(20) channels
        if i >= 20:
            high_20 = prices['high'].iloc[i-20:i].max()
            low_20 = prices['low'].iloc[i-20:i].min()
        else:
            high_20 = prices['high'].iloc[:i+1].max()
            low_20 = prices['low'].iloc[:i+1].min()
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long: price breaks above 20-period high with volume and trend
            if price > high_20 and volume_ok and close_1d.iloc[-1] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low with volume and trend
            elif price < low_20 and volume_ok and close_1d.iloc[-1] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 20-period low or trend changes
            if price < low_20 or close_1d.iloc[-1] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 20-period high or trend changes
            if price > high_20 or close_1d.iloc[-1] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0