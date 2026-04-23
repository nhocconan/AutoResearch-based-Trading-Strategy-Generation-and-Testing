#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND 12h EMA50 uptrend AND volume > 1.8x 30-period MA.
Short when price breaks below Donchian lower band AND 12h EMA50 downtrend AND volume > 1.8x 30-period MA.
Exit when price crosses Donchian middle band (20-period SMA) or opposite breakout.
Designed for ~30-40 trades/year with strong trend-following edge in both bull and bear markets.
Donchian channels adapt to volatility and provide robust breakout signals.
"""

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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) from 1h data? No, use same timeframe
    # For 4h timeframe, calculate 20-period Donchian on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    dc_upper = high_20
    dc_lower = low_20
    dc_middle = (dc_upper + dc_lower) / 2.0  # 20-period SMA
    
    # Calculate volume MA (30-period) for confirmation
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 30)  # need EMA50, Donchian20, volume MA30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or np.isnan(dc_middle[i]) or 
            np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 12h close > EMA50 = uptrend, close < EMA50 = downtrend
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        trend_up = close_12h_aligned[i] > ema_50_12h_aligned[i]
        trend_down = close_12h_aligned[i] < ema_50_12h_aligned[i]
        
        # Volume filter: 4h volume > 1.8x 30-period MA
        vol_filter = volume[i] > 1.8 * vol_ma_30[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > dc_upper[i]  # break above upper band
        breakout_down = close[i] < dc_lower[i]  # break below lower band
        middle_cross = (position == 1 and close[i] < dc_middle[i]) or \
                       (position == -1 and close[i] > dc_middle[i])
        opposite_breakout = (position == 1 and breakout_down) or \
                            (position == -1 and breakout_up)
        
        if position == 0:
            # Long: Donchian breakout up AND uptrend AND volume confirmation
            if breakout_up and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND downtrend AND volume confirmation
            elif breakout_down and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: middle band cross or opposite breakout
            exit_signal = middle_cross or opposite_breakout
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_12hEMA50_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0