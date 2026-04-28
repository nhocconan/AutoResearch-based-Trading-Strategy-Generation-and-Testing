#!/usr/bin/env python3
# Hypothesis: 1h price action within 4h/1d trend channels with volume confirmation.
# Uses 4h Donchian channels (20-period) for trend direction and 1d EMA(50) for long-term trend filter.
# Entry occurs when price pulls back to the 4h Donchian middle (mean) during uptrend/downtrend
# with volume > 1.3x 20-period average. Time-restricted to 08-20 UTC to avoid low-volume Asian session.
# Designed for 1h timeframe with ~80-120 total trades over 4 years to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel (trend direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channel (20-period)
    donch_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_mid_20 = (donch_high_20 + donch_low_20) / 2.0
    
    # Align 4h Donchian to 1h
    donch_high_20_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    donch_mid_20_aligned = align_htf_to_ltf(prices, df_4h, donch_mid_20)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    # Session filter: 08-20 UTC (active London/NY overlap)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Wait for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or
            np.isnan(donch_mid_20_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_ma[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Trend filters: price relative to 4h Donchian and 1d EMA
        uptrend_4h = close[i] > donch_mid_20_aligned[i]
        downtrend_4h = close[i] < donch_mid_20_aligned[i]
        uptrend_1d = close[i] > ema_50_1d_aligned[i]
        downtrend_1d = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: pullback to Donchian middle in trend direction with volume
        long_entry = uptrend_4h and uptrend_1d and (close[i] <= donch_mid_20_aligned[i] * 1.001) and volume_confirm[i]
        short_entry = downtrend_4h and downtrend_1d and (close[i] >= donch_mid_20_aligned[i] * 0.999) and volume_confirm[i]
        
        # Exit conditions: trend reversal or opposite Donchian breach
        long_exit = (not uptrend_4h) or (not uptrend_1d) or (close[i] >= donch_high_20_aligned[i])
        short_exit = (not downtrend_4h) or (not downtrend_1d) or (close[i] <= donch_low_20_aligned[i])
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Donchian_Pullback_4hTrend_1dEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0