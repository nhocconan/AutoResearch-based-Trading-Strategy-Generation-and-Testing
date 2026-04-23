#!/usr/bin/env python3
"""
Hypothesis: Daily Donchian(20) breakout with 1-week EMA50 trend filter and volume confirmation.
Long when price breaks above upper Donchian channel AND weekly EMA50 rising AND daily volume > 1.5x 20-day volume MA.
Short when price breaks below lower Donchian channel AND weekly EMA50 falling AND daily volume > 1.5x 20-day volume MA.
Exit when price touches opposite Donchian level or weekly EMA50 reverses.
Designed for 1d timeframe to capture medium-term trends while avoiding whipsaws in both bull and bear markets.
Weekly EMA50 provides robust trend filter, volume confirmation ensures momentum, Donchian channels adapt to volatility.
Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
"""

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
    
    # Calculate 1d Donchian channels (20-period)
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    for i in range(20, n):
        # Use lookback of 20 periods (current bar excluded for breakout)
        upper_channel[i] = np.max(high[i-20:i])
        lower_channel[i] = np.min(low[i-20:i])
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-day volume MA for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # Donchian (needs 20), EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = upper_channel[i]
        lower = lower_channel[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: daily volume > 1.5x 20-day volume MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Break above upper Donchian AND EMA50 rising AND volume filter
            if price > upper and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian AND EMA50 falling AND volume filter
            elif price < lower and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches lower Donchian (opposite) OR EMA50 starts falling
                if price < lower or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches upper Donchian (opposite) OR EMA50 starts rising
                if price > upper or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0