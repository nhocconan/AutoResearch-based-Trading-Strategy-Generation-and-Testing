#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA21 trend filter and volume spike confirmation.
Long when price breaks above Donchian upper(20) AND 12h EMA21 rising AND 15m volume > 2.0x 20-period MA.
Short when price breaks below Donchian lower(20) AND 12h EMA21 falling AND 15m volume > 2.0x 20-period MA.
Exit when price touches opposite Donchian level or 12h EMA21 reverses.
Uses 12h HTF for trend filter to capture longer-term direction and reduce whipsaw.
Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        donchian_upper[i] = np.max(high[i-lookback+1:i+1])
        donchian_lower[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate 12h EMA21 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Calculate 15m volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 21, 20)  # Donchian, EMA21, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_21_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        ema_val = ema_21_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate EMA21 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_21_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 15m volume > 2.0x 20-period MA
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Break above Donchian upper AND EMA21 rising AND volume filter
            if price > upper and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND EMA21 falling AND volume filter
            elif price < lower and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches lower Donchian OR EMA21 starts falling
                if price < lower or (i >= start_idx + 1 and ema_val < ema_21_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches upper Donchian OR EMA21 starts rising
                if price > upper or (i >= start_idx + 1 and ema_val > ema_21_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_12hEMA21_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0