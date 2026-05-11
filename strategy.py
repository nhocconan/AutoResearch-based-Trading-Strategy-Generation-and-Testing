#!/usr/bin/env python3
name = "4h_KC_Breakout_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR calculation (KC) and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for Keltner Channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Keltner Channels: EMA(20) ± 2*ATR
    ema_20 = pd.Series(close_1d).ewm(span=20, min_periods=20).mean().values
    kc_upper = ema_20 + 2 * atr_14
    kc_lower = ema_20 - 2 * atr_14
    
    # Align KC levels to 4h timeframe
    kc_upper_4h = align_htf_to_ltf(prices, df_1d, kc_upper)
    kc_lower_4h = align_htf_to_ltf(prices, df_1d, kc_lower)
    
    # 1d EMA50 for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume filter: current volume > 1.8x 24-period average (moderate threshold)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kc_upper_4h[i]) or np.isnan(kc_lower_4h[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above KC Upper AND above EMA50 (uptrend) AND volume filter
            if close[i] > kc_upper_4h[i] and close[i] > ema_50_4h[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below KC Lower AND below EMA50 (downtrend) AND volume filter
            elif close[i] < kc_lower_4h[i] and close[i] < ema_50_4h[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below KC Lower OR below EMA50 (trend change)
            if close[i] < kc_lower_4h[i] or close[i] < ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above KC Upper OR above EMA50 (trend change)
            if close[i] > kc_upper_4h[i] or close[i] > ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals