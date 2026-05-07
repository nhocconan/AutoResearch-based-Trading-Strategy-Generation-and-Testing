#!/usr/bin/env python3

# 1h_PowerTrend_4h1d_Confirmation_VolumeSpike
# Hypothesis: 1h trend following with 4h EMA50 and 1d EMA200 trend filters, plus volume spike confirmation.
# Uses 4h/1d for signal direction, 1h for entry timing. Designed to work in both bull and bear markets by
# requiring strong multi-timeframe alignment. Targets 15-30 trades/year to avoid fee drag.

name = "1h_PowerTrend_4h1d_Confirmation_VolumeSpike"
timeframe = "1h"
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
    
    # Get 4h and 1d data for trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 200:
        return np.zeros(n)
    
    # 4h EMA50 for intermediate trend
    close_4h = df_4h['close'].values
    ema_50_4h = np.full_like(close_4h, np.nan)
    for i in range(50, len(close_4h)):
        ema_50_4h[i] = np.mean(close_4h[i-50:i])  # Simple MA for robustness
    
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA200 for long-term trend
    close_1d = df_1d['close'].values
    ema_200_1d = np.full_like(close_1d, np.nan)
    for i in range(200, len(close_1d)):
        ema_200_1d[i] = np.mean(close_1d[i-200:i])  # Simple MA for robustness
    
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (2.0 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 12  # Prevent overtrading (approx 6 hours)
    
    start_idx = max(20, 50, 200)  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction using aligned EMAs
        trend_4h_up = close[i] > ema_50_4h_aligned[i]
        trend_4h_down = close[i] < ema_50_4h_aligned[i]
        trend_1d_up = close[i] > ema_200_1d_aligned[i]
        trend_1d_down = close[i] < ema_200_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars and in_session[i]:
            # Long: price above both 4h EMA50 and 1d EMA200 with volume spike
            if (trend_4h_up and trend_1d_up and vol_spike[i]):
                signals[i] = 0.20
                position = 1
                bars_since_last_trade = 0
            # Short: price below both 4h EMA50 and 1d EMA200 with volume spike
            elif (trend_4h_down and trend_1d_down and vol_spike[i]):
                signals[i] = -0.20
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit long: price crosses below 4h EMA50 OR 1d EMA200
            if (close[i] < ema_50_4h_aligned[i]) or (close[i] < ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 4h EMA50 OR 1d EMA200
            if (close[i] > ema_50_4h_aligned[i]) or (close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.20
    
    return signals