#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining 1-week Donchian breakouts with daily volume confirmation and ADX trend filter
# Weekly Donchian channels (20-period) capture major trend breakouts
# Daily volume > 1.5x 20-period average confirms breakout strength
# ADX > 25 ensures trading only in trending markets, avoiding whipsaws in ranges
# Works in bull markets (breakouts up) and bear markets (breakouts down)
# Target: 15-40 trades/year (60-160 total over 4 years) with 0.25 position sizing

name = "4h_WeeklyDonchian20_Volume_ADXTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Donchian channels (20-period) ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly high and low for Donchian channels
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Donchian channels: 20-period high and low
    high_series = pd.Series(weekly_high)
    low_series = pd.Series(weekly_low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Daily volume confirmation: >1.5x 20-period average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    daily_volume = df_1d['volume'].values
    vol_series = pd.Series(daily_volume)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * vol_ma_20
    volume_filter = align_htf_to_ltf(prices, df_1d, daily_volume > volume_threshold)
    
    # ADX trend filter (14-period) on daily timeframe
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth TR and DM (Wilder's smoothing)
    def WilderSmooth(data, period):
        smoothed = np.zeros_like(data)
        smoothed[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            smoothed[i] = (smoothed[i-1] * (period-1) + data[i]) / period
        return smoothed
    
    atr = WilderSmooth(tr, 14)
    plus_di = 100 * WilderSmooth(plus_dm, 14) / atr
    minus_di = 100 * WilderSmooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = WilderSmooth(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(adx_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly Donchian high with volume confirmation and ADX > 25
            if close[i] > donchian_high_aligned[i] and volume_filter[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below weekly Donchian low with volume confirmation and ADX > 25
            elif close[i] < donchian_low_aligned[i] and volume_filter[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low (trend reversal)
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high (trend reversal)
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals