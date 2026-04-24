#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA34 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA34 trend direction and volume spike confirmation.
- Donchian(20): Upper/lower bands from 20-period high/low on 4h.
- Trend filter: 12h EMA34 - price > 0 for uptrend, < 0 for downtrend.
- Volume confirmation: 4h volume > 1.5 * 20-period average volume.
- Entry: Long when price > Upper Band AND uptrend AND volume confirmation.
         Short when price < Lower Band AND downtrend AND volume confirmation.
- Exit: Opposite Donchian breakout (price < Upper Band for long exit, price > Lower Band for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by only trading breakouts with trend alignment, avoiding whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h_34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 12h price - EMA34 for trend direction
    trend_12h = close_12h - ema_12h_34
    
    # Align 12h trend to 4h timeframe
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Calculate 12h volume average for confirmation (20-period)
    if len(df_12h) < 20:
        return np.zeros(n)
    
    vol_ma_20_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate 4h Donchian(20) bands
    donchian_window = 20
    upper_band = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_band = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Calculate 4h volume average for confirmation (20-period)
    if n < 20:
        return np.zeros(n)
    
    vol_ma_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_window, 34)  # Need 20 for Donchian, 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i]) or
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(vol_ma_20_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: uptrend if trend_12h > 0, downtrend if trend_12h < 0
        uptrend = trend_12h_aligned[i] > 0
        downtrend = trend_12h_aligned[i] < 0
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume (4h)
        volume_confirm = curr_volume > 1.5 * vol_ma_20_4h[i]
        
        # Exit conditions: opposite Donchian breakout
        if position != 0:
            # Exit long: price < Upper Band
            if position == 1:
                if curr_close < upper_band[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > Lower Band
            elif position == -1:
                if curr_close > lower_band[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend and volume filters
        if position == 0:
            # Long: price > Upper Band AND uptrend AND volume confirmation
            long_condition = (curr_close > upper_band[i] and 
                            uptrend and
                            volume_confirm)
            
            # Short: price < Lower Band AND downtrend AND volume confirmation
            short_condition = (curr_close < lower_band[i] and 
                             downtrend and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA34Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0