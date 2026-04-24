#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA34 trend direction.
- Donchian channels: upper/lower bands based on 20-period high/low.
- Entry: Long when price breaks above Donchian upper AND price > 1d EMA34 AND volume > 2.0 * 20-period average volume.
         Short when price breaks below Donchian lower AND price < 1d EMA34 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Donchian breakout (price crosses below upper band for long exit, above lower band for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian breakouts capture strong momentum moves; volume spike confirms institutional participation; EMA34 filter avoids counter-trend trades in ranging markets.
- Works in bull markets (upside breakouts with trend) and bear markets (downside breakouts with trend) by only trading in direction of higher timeframe trend.
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 4h Donchian channels (20-period)
    # For Donchian we need to resample to 4h, calculate, then align back
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Donchian upper = 20-period high, lower = 20-period low
    donch_upper_4h = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    donch_lower_4h = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h timeframe (they're already 4h, just need alignment)
    donch_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_upper_4h)
    donch_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_lower_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Need 20 for Donchian/volume MA, 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(donch_upper_4h_aligned[i]) or 
            np.isnan(donch_lower_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian breakout
        if position != 0:
            # Exit long: price crosses below Donchian upper band
            if position == 1:
                if curr_close < donch_upper_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price crosses above Donchian lower band
            elif position == -1:
                if curr_close > donch_lower_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 2.0 * 20-period average volume
            volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
            
            # Long: price breaks above Donchian upper AND price > 1d EMA34
            long_condition = (curr_close > donch_upper_4h_aligned[i] and 
                            curr_close > ema34_1d_aligned[i] and
                            volume_confirm)
            
            # Short: price breaks below Donchian lower AND price < 1d EMA34
            short_condition = (curr_close < donch_lower_4h_aligned[i] and 
                             curr_close < ema34_1d_aligned[i] and
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

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0