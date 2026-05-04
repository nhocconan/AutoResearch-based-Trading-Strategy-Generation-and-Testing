#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter and weekly pivot confirmation
# Uses Donchian channel breakout from 6h for entry timing, aligned with 1d EMA50 trend
# Weekly Camarilla R4/S4 levels act as strong support/resistance for continuation
# Volume confirmation (>1.5x 20 EMA volume) filters false breakouts in low volatility
# Discrete sizing 0.25 targets 50-150 trades over 4 years to minimize fee drag
# Works in bull markets (breakout above R4 with uptrend) and bear markets (breakdown below S4 with downtrend)
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias)

name = "6h_Donchian20_1dEMA50_1wCamarillaR4S4_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for weekly Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d EMA(50) trend filter from prior completed 1d bar
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_shifted = np.roll(ema_50_1d, 1)
    ema_50_1d_shifted[0] = np.nan
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_shifted)
    
    # Calculate weekly Camarilla levels (R4, S4) from prior completed 1w bar
    # Camarilla: R4 = close + 1.1*(high-low), S4 = close - 1.1*(high-low)
    camarilla_range_1w = high_1w - low_1w
    r4_level_1w = close_1w + 1.1 * camarilla_range_1w
    s4_level_1w = close_1w - 1.1 * camarilla_range_1w
    
    # Shift to use prior completed 1w bar
    r4_shifted_1w = np.roll(r4_level_1w, 1)
    s4_shifted_1w = np.roll(s4_level_1w, 1)
    r4_shifted_1w[0] = np.nan
    s4_shifted_1w[0] = np.nan
    
    # Align to 6h timeframe
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_shifted_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_shifted_1w)
    
    # Calculate 6h Donchian(20) channels
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r4_1w_aligned[i]) or 
            np.isnan(s4_1w_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Donchian breakout above upper band AND price > 1d EMA50 AND price > weekly R4 AND volume spike
            if (close[i] > high_max_20[i] and close[i] > ema_50_1d_aligned[i] and 
                close[i] > r4_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: Donchian breakdown below lower band AND price < 1d EMA50 AND price < weekly S4 AND volume spike
            elif (close[i] < low_min_20[i] and close[i] < ema_50_1d_aligned[i] and 
                  close[i] < s4_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Donchian breakdown below lower band OR price crosses below 1d EMA50
            if close[i] < low_min_20[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Donchian breakout above upper band OR price crosses above 1d EMA50
            if close[i] > high_max_20[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals