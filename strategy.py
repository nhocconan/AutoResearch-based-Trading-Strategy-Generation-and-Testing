#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX regime filter and volume confirmation.
# Uses Donchian channels for breakout detection, 1d ADX > 25 to confirm trending regime,
# and volume spike (>1.5x 20-bar average) for confirmation. Designed to capture strong
# trends while avoiding choppy markets. Works in both bull and bear markets by following
# the breakout direction with trend filter. Target: 50-150 total trades over 4 years (12-37/year).
# Size: 0.25.

name = "4h_Donchian20_Breakout_1dADX_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX (regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smooth(tr, 14)
    plus_di_1d = 100 * wilders_smooth(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smooth(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smooth(dx_1d, 14)
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 4h Donchian(20) channels
    def donchian_channels(high, low, window):
        upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channels(high, low, 20)
    
    # 4h volume spike: >1.5x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # ADX needs ~50 bars for stability, Donchian needs 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or
            np.isnan(upper_20[i]) or
            np.isnan(lower_20[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 1d ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        # Breakout conditions
        long_breakout = close[i] > upper_20[i]
        short_breakout = close[i] < lower_20[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = trending and long_breakout and vol_confirm
        short_entry = trending and short_breakout and vol_confirm
        
        # Exit: opposite Donchian level (full reversion)
        long_exit = close[i] < lower_20[i]
        short_exit = close[i] > upper_20[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals