#!/usr/bin/env python3
"""
1h_4h_1d_HighLowBreakout_Volume
Hypothesis: Use 4h trend (price > 4h EMA50) and 1d momentum (close > open) for bias.
On 1h, enter long when price breaks above 4h high of prior day with volume spike.
Enter short when price breaks below 4h low of prior day with volume spike.
Exit on trend reversal or volume drop.
Designed for 1h timeframe with 4h/1d filters to limit trades to ~15-30/year.
Works in bull markets by buying strength and in bear markets by selling weakness.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    ema = np.zeros_like(close)
    if len(close) >= period:
        ema[period-1] = np.mean(close[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(close)):
            ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data once for trend and daily high/low
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h EMA50 for trend filter
    ema50_4h = calculate_ema(close_4h, 50)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 4h previous day high and low (using 4h bars)
    # Each day = 6 bars of 4h (24h / 4h = 6)
    prev_day_high = np.full_like(high_4h, np.nan)
    prev_day_low = np.full_like(low_4h, np.nan)
    
    for i in range(6, len(high_4h)):
        prev_day_high[i] = np.max(high_4h[i-6:i])
        prev_day_low[i] = np.min(low_4h[i-6:i])
    
    prev_day_high_aligned = align_htf_to_ltf(prices, df_4h, prev_day_high)
    prev_day_low_aligned = align_htf_to_ltf(prices, df_4h, prev_day_low)
    
    # Load 1d data for daily momentum (close > open)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    daily_momentum = close_1d > open_1d  # bullish day
    daily_momentum_aligned = align_htf_to_ltf(prices, df_1d, daily_momentum.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(prev_day_high_aligned[i]) or 
            np.isnan(prev_day_low_aligned[i]) or np.isnan(daily_momentum_aligned[i])):
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
        
        # Volume filter: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: uptrend + bullish day + break above prior day high + volume
            if (price > ema50_4h_aligned[i] and 
                daily_momentum_aligned[i] > 0.5 and  # bullish day
                price > prev_day_high_aligned[i] and 
                volume_ok):
                signals[i] = 0.20
                position = 1
            # Short conditions: downtrend + bearish day + break below prior day low + volume
            elif (price < ema50_4h_aligned[i] and 
                  daily_momentum_aligned[i] < 0.5 and  # bearish day
                  price < prev_day_low_aligned[i] and 
                  volume_ok):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or bearish day
            if price < ema50_4h_aligned[i] or daily_momentum_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend reversal or bullish day
            if price > ema50_4h_aligned[i] or daily_momentum_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_HighLowBreakout_Volume"
timeframe = "1h"
leverage = 1.0