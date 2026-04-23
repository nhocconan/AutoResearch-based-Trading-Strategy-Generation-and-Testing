#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above Donchian upper AND 1d EMA34 rising AND volume > 2.0x 20-period MA.
Short when price breaks below Donchian lower AND 1d EMA34 falling AND volume > 2.0x 20-period MA.
Exit when price crosses the Donchian midpoint or EMA34 reverses.
Uses 1d HTF for trend filter to avoid counter-trend trades in bear markets (2025+).
Volume spike filters for momentum confirmation to reduce false breakouts.
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        donchian_high[i] = np.max(high[i-lookback+1:i+1])
        donchian_low[i] = np.min(low[i-lookback+1:i+1])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # Calculate 4h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 20)  # Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_1d_aligned[i-1]
            ema_rising = ema_34_1d_aligned[i] > ema_prev
            ema_falling = ema_34_1d_aligned[i] < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 4h volume > 2.0x 20-period MA (high threshold to reduce trades)
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND EMA34 rising AND volume filter
            if close[i] > donchian_high[i] and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND EMA34 falling AND volume filter
            elif close[i] < donchian_low[i] and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price crosses below Donchian midpoint OR EMA34 starts falling
                if close[i] < donchian_mid[i] or (i >= start_idx + 1 and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price crosses above Donchian midpoint OR EMA34 starts rising
                if close[i] > donchian_mid[i] or (i >= start_idx + 1 and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0