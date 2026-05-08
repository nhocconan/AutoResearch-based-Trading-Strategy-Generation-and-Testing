#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly ADX filter with daily Donchian breakout and volume confirmation
# Uses weekly trend strength to filter daily breakouts, reducing false signals in choppy markets.
# ADX > 25 on weekly indicates strong trend, then we look for daily Donchian(20) breakouts
# in the direction of the weekly trend. Volume must confirm breakout.
# Designed for low frequency and high edge in both bull and bear markets.
# Target: 30-100 total trades over 4 years (7-25/year)

name = "1d_Donchian_WeeklyADX_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    # Calculate weekly ADX(14) for trend strength
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # True Range
    tr1 = high_w[1:] - low_w[1:]
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_w[1:] - high_w[:-1]
    down_move = low_w[:-1] - low_w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di14 = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 0)
    adx14 = wilders_smoothing(dx, 14)
    
    adx_weekly = align_htf_to_ltf(prices, df_weekly, adx14)
    
    # Get daily data for Donchian channels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    
    # Daily Donchian(20)
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
    
    donchian_high = rolling_max(high_d, 20)
    donchian_low = rolling_min(low_d, 20)
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_daily, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_daily, donchian_low)
    
    # Daily trend: EMA(50) for filtering
    close_d = df_daily['close'].values
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_d_aligned = align_htf_to_ltf(prices, df_daily, ema50_d)
    
    # Volume spike: current > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_weekly[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(ema50_d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_weekly[i]
        dc_high = donchian_high_aligned[i]
        dc_low = donchian_low_aligned[i]
        ema50_val = ema50_d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Strong trend filter: ADX > 25
            if adx_val > 25:
                # Long: price breaks above Donchian high + above EMA50 + volume spike
                if (close[i] > dc_high and close[i] > ema50_val and vol_spike):
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low + below EMA50 + volume spike
                elif (close[i] < dc_low and close[i] < ema50_val and vol_spike):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR ADX weakens
            if close[i] < donchian_low_aligned[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high OR ADX weakens
            if close[i] > donchian_high_aligned[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals