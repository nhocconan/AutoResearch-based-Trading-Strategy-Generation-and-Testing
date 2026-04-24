#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA50 trend direction and volume spike detection.
- Donchian(20): Upper/lower bands from 20-period high/low on 4h.
- Trend filter: 12h EMA50 slope > 0 for uptrend, < 0 for downtrend.
- Volume confirmation: 4h volume > 2.0 * 20-period average volume.
- Entry: Long when price > Upper Band AND uptrend AND volume spike.
         Short when price < Lower Band AND downtrend AND volume spike.
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # EMA50 calculation
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # EMA50 slope for trend direction (positive = uptrend, negative = downtrend)
    ema50_slope = np.diff(ema50_12h, prepend=ema50_12h[0])
    
    # Align EMA50 slope to 4h timeframe
    ema50_slope_aligned = align_htf_to_ltf(prices, df_12h, ema50_slope)
    
    # Calculate 12h volume average for spike detection (20-period)
    if len(df_12h) < 20:
        return np.zeros(n)
    
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate 4h Donchian(20) bands
    donchian_window = 20
    upper_band = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_band = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_window, 50)  # Need 20 for Donchian, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_slope_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i]) or
            np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: EMA50 slope > 0 for uptrend, < 0 for downtrend
        uptrend = ema50_slope_aligned[i] > 0
        downtrend = ema50_slope_aligned[i] < 0
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_12h_aligned[i] if not np.isnan(vol_ma_20_12h_aligned[i]) else False
        
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

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0