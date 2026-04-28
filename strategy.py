#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX(14) trend filter and volume confirmation.
# Enter long when price breaks above Donchian upper channel, 1d ADX > 25 (trending), and volume > 1.5x 20-bar average.
# Enter short when price breaks below Donchian lower channel, 1d ADX > 25, and volume > 1.5x 20-bar average.
# Exit when price touches the opposite Donchian channel or crosses the 20-bar EMA.
# Uses discrete position sizing (0.25) to minimize fee drag while maintaining edge.
# Target: 80-150 total trades over 4 years (20-38/year) to avoid excessive fee churn.
# Donchian channels provide objective breakout levels; 1d ADX filters for daily trend strength;
# Volume confirmation ensures breakout participation.

name = "4h_DonchianBreakout_1dADX25_VolumeConfirm_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian channels (20-period) on 4h data
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_upper = rolling_max(high, 20)
    donchian_lower = rolling_min(low, 20)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    # 20-bar EMA for exit
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d ADX trend filter: >25 indicates trending market
        adx_trend = adx_aligned[i] > 25
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_upper[i]
        breakout_down = close[i] < donchian_lower[i]
        
        # Exit conditions: price touches opposite Donchian channel or crosses 20-bar EMA
        exit_long = close[i] < donchian_lower[i] or close[i] < ema_20[i]
        exit_short = close[i] > donchian_upper[i] or close[i] > ema_20[i]
        
        # Handle entries and exits
        if breakout_up and adx_trend and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_down and adx_trend and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
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