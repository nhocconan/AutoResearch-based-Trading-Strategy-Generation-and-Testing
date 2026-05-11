#!/usr/bin/env python3
# 1d_Donchian20_Breakout_1wTrend_Volume_Confirm_v1
# Hypothesis: 1-day Donchian(20) breakout with 1-week EMA50 trend filter and volume confirmation.
# Uses 1d timeframe to reduce trade frequency (target 10-25 trades/year). The 1-week EMA50
# filter ensures we only trade in the direction of the higher-timeframe trend, improving
# performance in both bull and bear markets. Volume confirmation adds conviction to breakouts.
# Designed for BTC/ETH robustness with controlled trade frequency to minimize fee drag.

name = "1d_Donchian20_Breakout_1wTrend_Volume_Confirm_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # === 1d Data (loaded ONCE) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Donchian Channel (20-period) ===
    # Highest high and lowest low over past 20 days
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # === 1w Data (loaded ONCE) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === 1w EMA50 Trend Filter ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # Require 1.5x average volume
    
    # === Signal Parameters ===
    position_size = 0.25  # 25% of capital per trade
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    holding_bars = 0
    
    # Start after warmup (covers Donchian and EMA50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema50_1w_1d[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                holding_bars = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above 20-day high + above 1w EMA50 + volume spike
            if (high[i] > high_20[i] and close[i] > ema50_1w_1d[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
                holding_bars = 0
            # Short: Break below 20-day low + below 1w EMA50 + volume spike
            elif (low[i] < low_20[i] and close[i] < ema50_1w_1d[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
                holding_bars = 0
        else:
            # Enforce minimum holding period (5 days)
            holding_bars += 1
            if holding_bars < 5:
                signals[i] = position_size if position == 1 else -position_size
                continue
            
            # Exit: Price closes back inside the Donchian channel
            if position == 1:
                if close[i] < low_20[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > high_20[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = -position_size
    
    return signals