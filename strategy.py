#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1w Donchian channel breakout with 1d ADX trend filter and volume confirmation.
- 1w Donchian(20) breakout captures major trends
- 1d ADX > 25 filters for trending markets only (avoids whipsaws in ranges)
- Volume confirmation > 1.5x average reduces false breakouts
- Position size: 0.25 discrete level to minimize fee churn
- Target: 12-37 trades/year (50-150 over 4 years) to avoid fee drag
- Works in bull/bear via ADX trend filter (only trades when trending)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Volume confirmation: > 1.5x 24-period average (strict for 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # 1w Donchian channel (20-period)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian bands
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    
    # 1d ADX for trend filter (ADX > 25 = trending)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 24, 20)  # ADX, volume MA, Donchian
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(upper_20_aligned[i]) or
            np.isnan(lower_20_aligned[i]) or
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 25
        trending = adx_aligned[i] > 25
        
        # Donchian breakout signals
        breakout_up = close[i] > upper_20_aligned[i-1]  # Close above prior 1w upper band
        breakout_down = close[i] < lower_20_aligned[i-1]  # Close below prior 1w lower band
        
        if position == 0:
            # Long: 1w Donchian upper breakout AND trending AND volume confirmation AND in session
            if breakout_up and trending and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: 1w Donchian lower breakdown AND trending AND volume confirmation AND in session
            elif breakout_down and trending and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian lower band break OR ADX < 20 (trend weakening)
            if close[i] < lower_20_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian upper band break OR ADX < 20 (trend weakening)
            if close[i] > upper_20_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dADX25_VolumeSpike_Session"
timeframe = "12h"
leverage = 1.0