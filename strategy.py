#!/usr/bin/env python3
"""
4h_1dDonchian20_12hTrend_Volume
Hypothesis: Daily Donchian channel breakouts with 12-hour trend filter and volume confirmation work in both bull and bear markets.
The daily Donchian(20) captures significant breakouts, while the 12-hour EMA50 filters for trend alignment.
Volume confirmation ensures breakouts have conviction. This approach reduces trade frequency by using higher timeframe signals
and avoids overtrading by requiring confluence of trend, breakout, and volume.
"""

name = "4h_1dDonchian20_12hTrend_Volume"
timeframe = "4h"
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
    
    # === Daily Donchian Channel (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels from daily high/low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: highest high over past 20 days
    upper_20 = np.full_like(high_1d, np.nan)
    for i in range(20, len(high_1d)):
        upper_20[i] = np.max(high_1d[i-20:i])
    
    # Lower band: lowest low over past 20 days
    lower_20 = np.full_like(low_1d, np.nan)
    for i in range(20, len(low_1d)):
        lower_20[i] = np.min(low_1d[i-20:i])
    
    # Align to 4h timeframe (only available after daily candle closes)
    upper_20_4h = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_4h = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # === 12-hour Trend Filter (EMA50) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = np.full_like(close_12h, np.nan)
    close_12h_series = pd.Series(close_12h)
    ema50_values = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema50_12h_4h = align_htf_to_ltf(prices, df_12h, ema50_values)
    
    # === Volume Filter (1.5x 20-period EMA on 4h) ===
    vol_ema20 = np.full_like(volume, np.nan)
    volume_series = pd.Series(volume)
    vol_ema20_values = volume_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20_values * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers all indicator calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_20_4h[i]) or np.isnan(lower_20_4h[i]) or 
            np.isnan(ema50_12h_4h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above daily upper band with uptrend and volume
            if (close[i] > upper_20_4h[i] and 
                close[i] > ema50_12h_4h[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below daily lower band with downtrend and volume
            elif (close[i] < lower_20_4h[i] and 
                  close[i] < ema50_12h_4h[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below daily lower band (mean reversion)
            if close[i] < lower_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above daily upper band (mean reversion)
            if close[i] > upper_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals