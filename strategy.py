#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX(14) trend filter and volume confirmation.
# Long when price breaks above 6h Donchian upper band AND 1d ADX > 25 (strong trend) AND 6h volume > 2.0x 20-period volume MA.
# Short when price breaks below 6h Donchian lower band AND 1d ADX > 25 AND 6h volume > 2.0x 20-period volume MA.
# Exit on retracement to 6h Donchian middle band or ADX < 20 (weak trend).
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Donchian channels provide clear breakout levels, 1d ADX filters for higher-timeframe trend strength, volume confirms participation.
# Works in both bull and bear markets by only trading breakouts when the 1d trend is strong (ADX > 25) and volume confirms.

name = "6h_Donchian20_1dADX25_VolumeSpike_Session"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend strength
    # ADX calculation requires +DI, -DI, and DX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(values[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1]/period) + values[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_period = 20
    upper_band = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_band = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    middle_band = (upper_band + lower_band) / 2
    
    # Calculate 6h volume 20-period MA for spike detection
    volume_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_aligned[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(middle_band[i]) or np.isnan(volume_ma_6h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 6h volume > 2.0x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_6h[i] * 2.0)
        
        # Donchian breakout conditions
        breakout_up = high_val > upper_band[i]  # Price breaks above upper band
        breakout_down = low_val < lower_band[i]  # Price breaks below lower band
        
        # 1d ADX trend strength conditions
        strong_trend = adx_aligned[i] > 25.0   # Strong trend
        weak_trend = adx_aligned[i] < 20.0     # Weak trend (exit condition)
        
        if position == 0:
            # Long: Donchian breakout up AND strong trend AND volume spike AND session
            if breakout_up and strong_trend and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND strong trend AND volume spike AND session
            elif breakout_down and strong_trend and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches middle band OR trend weakens
            if close_val < middle_band[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches middle band OR trend weakens
            if close_val > middle_band[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals