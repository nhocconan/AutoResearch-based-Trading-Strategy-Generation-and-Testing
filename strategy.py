#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and 1d volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA50 trend direction, 1d for volume spike filter (>2.0 * 20-period average volume).
- Donchian(20): Upper/lower bands from 20-period high/low.
- Trend filter: price > EMA50_12h for long bias, price < EMA50_12h for short bias.
- Volume confirmation: current 4h volume > 2.0 * 20-period average 1d volume (aligned).
- Entry: Long when price > Upper Band AND price > EMA50_12h AND volume confirmation.
         Short when price < Lower Band AND price < EMA50_12h AND volume confirmation.
- Exit: Opposite Donchian breakout (price < Upper Band for long exit, price > Lower Band for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by requiring volume confirmation to avoid false breakouts,
  and using 12h EMA50 to align with medium-term trend.
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
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 1d volume average for confirmation (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
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
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price relative to 12h EMA50
        long_bias = curr_close > ema50_12h_aligned[i]
        short_bias = curr_close < ema50_12h_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
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
            # Long: price > Upper Band AND long bias AND volume confirmation
            long_condition = (curr_close > upper_band[i] and 
                            long_bias and
                            volume_confirm)
            
            # Short: price < Lower Band AND short bias AND volume confirmation
            short_condition = (curr_close < lower_band[i] and 
                             short_bias and
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