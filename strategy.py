#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 12h ADX(14) > 25 trend filter + volume > 2.0x 20-bar avg
    # Enter long on breakout above Donchian high in strong uptrend (ADX > 25), short on breakout below Donchian low in strong downtrend
    # Exit when price crosses Donchian midpoint
    # Volume confirmation ensures breakout has participation
    # ADX filter avoids whipsaws in ranging markets
    # Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_window = 20
    donchian_high_4h = pd.Series(high_4h).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid_4h = (donchian_high_4h + donchian_low_4h) / 2.0
    
    # Align 4h Donchian levels to 4h timeframe (no-op but for consistency)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    
    # Get 12h data for ADX trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range for 12h
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = np.nan  # first value has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate +DM and -DM for 12h
    up_move = np.diff(high_12h, prepend=high_12h[0])
    down_move = np.diff(low_12h, prepend=low_12h[0]) * -1  # invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Calculate smoothed TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_12h = wilders_smoothing(tr, 14)
    plus_di_12h = 100 * wilders_smoothing(plus_dm, 14) / atr_12h
    minus_di_12h = 100 * wilders_smoothing(minus_dm, 14) / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = wilders_smoothing(dx_12h, 14)
    
    # Align 12h ADX values to 4h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Trend filter: ADX > 25 (strong trend)
    trend_filter = adx_12h_aligned > 25
    
    # Volume confirmation: volume > 2.0x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(donchian_window, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > donchian_high_aligned[i-1]  # break above previous Donchian high
        breakout_down = close[i] < donchian_low_aligned[i-1]  # break below previous Donchian low
        
        # Entry conditions with trend filter and volume confirmation
        long_entry = breakout_up and trend_filter[i] and volume_confirmed[i] and position != 1
        short_entry = breakout_down and trend_filter[i] and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < donchian_mid_aligned[i])
        exit_short = (position == -1 and close[i] > donchian_mid_aligned[i])
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_adx_volume_filter_v1"
timeframe = "4h"
leverage = 1.0