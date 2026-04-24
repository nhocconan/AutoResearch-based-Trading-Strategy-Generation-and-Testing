#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian breakout with 4h EMA50 trend filter, 1d volume spike confirmation, and session filter (08-20 UTC).
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA50 trend direction, 1d for volume spike confirmation (>2x 20-period average).
- Entry: Long when price breaks above Donchian(20) high AND price > 4h EMA50 AND 1d volume > 2 * 20-period average volume AND hour in [08,20] UTC.
         Short when price breaks below Donchian(20) low AND price < 4h EMA50 AND 1d volume > 2 * 20-period average volume AND hour in [08,20] UTC.
- Exit: Opposite Donchian breakout (price < Donchian(20) low for long exit, price > Donchian(20) high for short exit).
- Signal size: 0.20 discrete to minimize fee drag.
- Uses actual price structure (Donchian channels) with trend and volume filters to avoid whipsaws.
- Session filter reduces noise trades during low-liquidity hours.
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1h Donchian channels (20-period)
    if len(close) < 20:
        return np.zeros(n)
    
    # Rolling max/min for Donchian channels
    high_series = pd.Series(close)  # Use close for breakout (standard practice)
    low_series = pd.Series(close)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema50_4h = ema(df_4h['close'].values, 50)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1d volume average for confirmation (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if not in trading session (08-20 UTC)
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian breakout
        if position != 0:
            # Exit long: price < Donchian low
            if position == 1:
                if curr_close < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > Donchian high
            elif position == -1:
                if curr_close > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend filter, volume confirmation, and session
        if position == 0:
            # Volume confirmation: current 1d volume > 2 * 20-period average volume
            volume_confirm = not np.isnan(vol_ma_20_1d_aligned[i]) and curr_volume > 2.0 * vol_ma_20_1d_aligned[i]
            
            # Long: price > Donchian high AND price > 4h EMA50 AND volume confirmation
            long_condition = (curr_close > donchian_high[i] and 
                            curr_close > ema50_4h_aligned[i] and
                            volume_confirm)
            
            # Short: price < Donchian low AND price < 4h EMA50 AND volume confirmation
            short_condition = (curr_close < donchian_low[i] and 
                             curr_close < ema50_4h_aligned[i] and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Donchian_Breakout_4hEMA50_1dVolumeSpike_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0