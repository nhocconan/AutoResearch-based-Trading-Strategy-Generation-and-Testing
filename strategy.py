#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Choppiness Index for regime detection and 1d EMA crossover for trend
# - Uses 12h Choppiness Index to identify trending vs ranging markets
# - Uses 1d EMA crossover (8/21) to determine trend direction
# - Enters long when market is trending (CHOP < 38.2) and EMA 8 crosses above EMA 21
# - Enters short when market is trending (CHOP < 38.2) and EMA 8 crosses below EMA 21
# - Uses volume spike as confirmation for entry
# - Exits when market becomes ranging (CHOP > 61.8) or opposite EMA crossover occurs
# - Designed to capture trending moves while avoiding choppy markets
# - Target: 100-200 total trades over 4 years (25-50/year) with 0.25 position sizing

name = "4h_12hChop_1dEMA_Crossover"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Choppiness Index calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Get 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 12h Choppiness Index (14)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR using Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_12h = wilders_smoothing(tr, 14)
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    sum_atr = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)  # Avoid division by zero
    
    # Calculate 1d EMA crossover (8, 21)
    close_1d = df_1d['close'].values
    ema_8 = pd.Series(close_1d).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_21 = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # EMA crossover signals
    ema_cross_up = np.where((ema_8 > ema_21) & (np.roll(ema_8, 1) <= np.roll(ema_21, 1)), 1, 0)
    ema_cross_down = np.where((ema_8 < ema_21) & (np.roll(ema_8, 1) >= np.roll(ema_21, 1)), 1, 0)
    
    # Align 12h Choppiness Index to 4h timeframe
    chop_4h = align_htf_to_ltf(prices, df_12h, chop)
    
    # Align 1d EMA and crossover signals to 4h timeframe
    ema_8_4h = align_htf_to_ltf(prices, df_1d, ema_8)
    ema_21_4h = align_htf_to_ltf(prices, df_1d, ema_21)
    ema_cross_up_4h = align_htf_to_ltf(prices, df_1d, ema_cross_up)
    ema_cross_down_4h = align_htf_to_ltf(prices, df_1d, ema_cross_down)
    
    # Volume filters (4h timeframe)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (1.5 * vol_ma_10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(chop_4h[i]) or np.isnan(ema_8_4h[i]) or np.isnan(ema_21_4h[i]) or
            np.isnan(ema_cross_up_4h[i]) or np.isnan(ema_cross_down_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: trending market (CHOP < 38.2) + EMA bullish crossover + volume spike
            if chop_4h[i] < 38.2 and ema_cross_up_4h[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: trending market (CHOP < 38.2) + EMA bearish crossover + volume spike
            elif chop_4h[i] < 38.2 and ema_cross_down_4h[i] == 1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: market becomes ranging (CHOP > 61.8) OR EMA bearish crossover
            if chop_4h[i] > 61.8 or ema_cross_down_4h[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: market becomes ranging (CHOP > 61.8) OR EMA bullish crossover
            if chop_4h[i] > 61.8 or ema_cross_up_4h[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals