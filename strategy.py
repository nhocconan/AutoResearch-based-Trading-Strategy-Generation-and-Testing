#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm
Hypothesis: Daily Camarilla R1/S1 breakout with weekly trend filter and volume confirmation. 
Goes long when price breaks above R1 with weekly uptrend and above-average volume, 
short when price breaks below S1 with weekly downtrend and above-average volume.
Uses discrete sizing (0.25) to minimize fees. Target: 15-25 trades/year.
Works in bull via breakouts with trend, in bear via mean reversion at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for today using yesterday's OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    # R2 = close + 0.55*(high-low), R1 = close + 0.275*(high-low)
    # S1 = close - 0.275*(high-low), S2 = close - 0.55*(high-low), etc.
    # We only need R1 and S1 for breakout
    prev_close = np.concatenate([[close_1d[0]], close_1d[:-1]])  # yesterday's close
    prev_high = np.concatenate([[high_1d[0]], high_1d[:-1]])   # yesterday's high
    prev_low = np.concatenate([[low_1d[0]], low_1d[:-1]])     # yesterday's low
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 0.275 * camarilla_range
    s1 = prev_close - 0.275 * camarilla_range
    
    # Align Camarilla levels to 1d timeframe (no shift needed as we use yesterday's data)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: today's volume vs 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above R1, weekly uptrend (price > EMA50), volume above average
            long_signal = (close[i] > r1_aligned[i]) and (close[i] > ema_50_1w_aligned[i]) and (volume[i] > vol_ma_20[i])
            # Short: price breaks below S1, weekly downtrend (price < EMA50), volume above average
            short_signal = (close[i] < s1_aligned[i]) and (close[i] < ema_50_1w_aligned[i]) and (volume[i] > vol_ma_20[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price closes below S1 (mean reversion) or weekly trend turns down
            exit_signal = (close[i] < s1_aligned[i]) or (close[i] < ema_50_1w_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price closes above R1 (mean reversion) or weekly trend turns up
            exit_signal = (close[i] > r1_aligned[i]) or (close[i] > ema_50_1w_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0