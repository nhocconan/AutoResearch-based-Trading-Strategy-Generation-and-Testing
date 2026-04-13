#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w ADX regime filter
    # Long: price breaks above 4h upper band AND 1w ADX > 25 (strong trend) AND 1d volume > 2.0x 20-period avg
    # Short: price breaks below 4h lower band AND 1w ADX > 25 AND 1d volume > 2.0x 20-period avg
    # Exit: price retests breakout level (middle of channel) or opposite band touch
    # Using 4h timeframe for optimal trade frequency (target 20-50/year), Donchian for structure,
    # 1w ADX to avoid ranging markets, and 1d volume confirmation to filter breakouts.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly ADX(14) for trend strength filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: smoothed = (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # DI+ and DI-
    plus_di14 = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di14 = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align weekly ADX to 4h
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 4h Donchian channels (20-period)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    middle_band = np.full(n, np.nan)
    
    for i in range(20, n):
        upper_band[i] = np.max(high[i-20:i])
        lower_band[i] = np.min(low[i-20:i])
        middle_band[i] = (upper_band[i] + lower_band[i]) / 2
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Align daily data to 4h
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Calculate daily volume MA(20) for spike detection
    vol_ma_1d = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_1d[i] = np.mean(volume_1d_aligned[i-20:i])
    volume_spike = volume_1d_aligned > (2.0 * vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 indicates strong trending market
        strong_trend = adx_1w_aligned[i] > 25
        
        # Donchian breakout conditions
        breakout_upper = close[i] > upper_band[i]
        breakout_lower = close[i] < lower_band[i]
        
        # Exit conditions: retest middle band or touch opposite band
        retest_middle_long = close[i] < middle_band[i] and position == 1  # Long exit on middle band retest
        retest_middle_short = close[i] > middle_band[i] and position == -1  # Short exit on middle band retest
        touch_lower = close[i] < lower_band[i]  # Exit long on lower band touch
        touch_upper = close[i] > upper_band[i]  # Exit short on upper band touch
        
        # Entry logic: Donchian breakout + strong trend + volume confirmation
        long_entry = breakout_upper and strong_trend and volume_spike[i]
        short_entry = breakout_lower and strong_trend and volume_spike[i]
        
        # Exit logic: middle band retest or opposite band touch
        long_exit = retest_middle_long or touch_lower
        short_exit = retest_middle_short or touch_upper
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_1w_donchian_breakout_adx_volume_v1"
timeframe = "4h"
leverage = 1.0