#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 1d trend filter and volume confirmation
# Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures bull/bear strength.
# Combined with 1d EMA trend filter to avoid counter-trend trades and volume confirmation.
# Works in bull markets (buy strength) and bear markets (sell weakness).
# Target: 12-37 trades per year (50-150 total over 4 years) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (13-period EMA)
    def calculate_ema(data, period):
        ema = np.full(len(data), np.nan)
        if len(data) < period:
            return ema
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    ema13 = calculate_ema(close, 13)
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Calculate daily EMA21 trend filter
    close_1d = df_1d['close'].values
    ema21_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (21 + 1)
    ema21_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema21_1d[i] = (close_1d[i] - ema21_1d[i-1]) * ema_multiplier + ema21_1d[i-1]
    
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema21_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        bp = bull_power[i]
        br = bear_power[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        daily_ema = ema21_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Bull Power > 0 (strength) + above daily EMA + volume
            if bp > 0 and close[i] > daily_ema and volume_confirm:
                position = 1
                signals[i] = position_size
            # Short: Bear Power < 0 (weakness) + below daily EMA + volume
            elif br < 0 and close[i] < daily_ema and volume_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power turns negative or price crosses below daily EMA
            if bp <= 0 or close[i] < daily_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bear Power turns positive or price crosses above daily EMA
            if br >= 0 or close[i] > daily_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_elder_ray_trend_volume_v1"
timeframe = "6h"
leverage = 1.0