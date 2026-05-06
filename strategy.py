#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with 1w ADX(14) > 25 trend filter and volume confirmation
# Long when price breaks above 1d Donchian upper band AND 1w ADX > 25 (strong trend) AND volume > 1.3 * avg_volume(20) on 12h
# Short when price breaks below 1d Donchian lower band AND 1w ADX > 25 (strong trend) AND volume > 1.3 * avg_volume(20) on 12h
# Exit when price crosses back through the 1d Donchian midpoint (upper+lower)/2
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1d Donchian(20) provides strong breakout levels that reduce whipsaw in ranging markets
# 1w ADX > 25 ensures we only trade in strong trending conditions (works in both bull and bear)
# Volume confirmation (1.3x) validates breakout strength while limiting overtrading

name = "12h_1dDonchian20_1wADX25_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed 1d bars for Donchian(20)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_h_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_l_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_m_1d = (donchian_h_1d + donchian_l_1d) / 2.0
    
    # Align 1d Donchian to 12h timeframe (wait for completed 1d bar)
    donchian_h_aligned = align_htf_to_ltf(prices, df_1d, donchian_h_1d)
    donchian_l_aligned = align_htf_to_ltf(prices, df_1d, donchian_l_1d)
    donchian_m_aligned = align_htf_to_ltf(prices, df_1d, donchian_m_1d)
    
    # Get 1w data ONCE before loop for ADX(14) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:  # Need at least 14 completed weekly bars for ADX(14)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX(14)
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    plus_di_1w = 100 * wilders_smoothing(plus_dm, 14) / atr_1w
    minus_di_1w = 100 * wilders_smoothing(minus_dm, 14) / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = wilders_smoothing(dx_1w, 14)
    
    # Align 1w ADX to 12h timeframe (wait for completed 1w bar)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_h_aligned[i]) or np.isnan(donchian_l_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper band, 1w ADX > 25 (strong trend), volume confirmation, in session
            if (close[i] > donchian_h_aligned[i] and 
                adx_1w_aligned[i] > 25.0 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower band, 1w ADX > 25 (strong trend), volume confirmation, in session
            elif (close[i] < donchian_l_aligned[i] and 
                  adx_1w_aligned[i] > 25.0 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1d Donchian midpoint
            if close[i] < donchian_m_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 1d Donchian midpoint
            if close[i] > donchian_m_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals