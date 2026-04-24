#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for EMA50 trend direction, 1d for Donchian calculation (based on daily high/low).
- Donchian: Upper = 20-period high, Lower = 20-period low on 1d data.
- Entry: Long when price > 1d Upper Band AND price > 1w EMA50 AND volume > 1.5 * 20-period average volume.
         Short when price < 1d Lower Band AND price < 1w EMA50 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Donchian break (price < 1d Lower for long exit, price > 1d Upper for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian breakouts capture momentum; 1w EMA50 filter avoids counter-trend trades in bear markets.
- Works in bull markets (strong upward breaks) and bear markets (strong downward breaks) with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for Donchian
        return np.zeros(n)
    
    # 1d rolling high/low for Donchian channels
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian/volume MA, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian break
        if position != 0:
            # Exit long: price < 1d Lower Band
            if position == 1:
                if curr_close < lower_band_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > 1d Upper Band
            elif position == -1:
                if curr_close > upper_band_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian break with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 1.5 * 20-period average volume
            volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
            
            # Long: price > 1d Upper Band AND price > 1w EMA50 AND volume confirmation
            long_condition = (curr_close > upper_band_aligned[i] and 
                            curr_close > ema50_1w_aligned[i] and
                            volume_confirm)
            
            # Short: price < 1d Lower Band AND price < 1w EMA50 AND volume confirmation
            short_condition = (curr_close < lower_band_aligned[i] and 
                             curr_close < ema50_1w_aligned[i] and
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

name = "12h_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0