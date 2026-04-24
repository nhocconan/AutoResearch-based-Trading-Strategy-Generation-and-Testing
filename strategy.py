#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend direction.
- Donchian(20): 20-period high/low on 1d data.
- Entry: Long when close > 20d high AND price > 1w EMA50 AND volume > 1.5 * 20-period average volume.
         Short when close < 20d low AND price < 1w EMA50 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Donchian signal (close < 20d low for long exit, close > 20d high for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian breakouts capture momentum; 1w EMA50 filter ensures trading with the weekly trend.
- Works in bull markets (breakouts above rising EMA50) and bear markets (breakouts below falling EMA50).
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
    
    # Calculate 1d Donchian(20) channels
    if n < 20:
        return np.zeros(n)
    
    high_rolling = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_rolling = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_rolling.values
    donchian_low = low_rolling.values
    
    # Calculate 1d volume average for confirmation (20-period)
    if n < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 20  # Need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if np.isnan(ema50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian signal
        if position != 0:
            # Exit long: close < 20d low
            if position == 1:
                if curr_close < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close > 20d high
            elif position == -1:
                if curr_close > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 1.5 * 20-period average volume
            volume_confirm = curr_volume > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
            
            # Long: close > 20d high AND price > 1w EMA50
            long_condition = (curr_close > donchian_high[i] and 
                            curr_close > ema50_1w_aligned[i] and
                            volume_confirm)
            
            # Short: close < 20d low AND price < 1w EMA50
            short_condition = (curr_close < donchian_low[i] and 
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

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0