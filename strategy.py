#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian breakout with weekly ADX trend filter and volume confirmation
# Long when price breaks above 20-day high + weekly ADX > 25 + daily volume > 1.5x 20-day average
# Short when price breaks below 20-day low + weekly ADX > 25 + daily volume > 1.5x 20-day average
# Exit when price crosses opposite 10-day level or weekly ADX falls below 20
# Uses Donchian channels for breakout signals, volume for confirmation, ADX for trend strength
# Targets 15-25 trades/year to minimize fee decay while capturing strong directional moves in bull/bear markets

name = "1d_Donchian_Breakout_WeeklyADX_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels and volume
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Get weekly data for ADX trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate weekly ADX(14) for trend strength
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # True Range
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high - np.roll(weekly_close, 1))
    tr3 = np.abs(weekly_low - np.roll(weekly_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((weekly_high - np.roll(weekly_high, 1)) > (np.roll(weekly_low, 1) - weekly_low), 
                       np.maximum(weekly_high - np.roll(weekly_high, 1), 0), 0)
    dm_minus = np.where((np.roll(weekly_low, 1) - weekly_low) > (weekly_high - np.roll(weekly_high, 1)), 
                        np.maximum(np.roll(weekly_low, 1) - weekly_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder smoothing)
    def smooth_series(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = smooth_series(tr, 14)
    dm_plus_smooth = smooth_series(dm_plus, 14)
    dm_minus_smooth = smooth_series(dm_minus, 14)
    
    # DI values
    di_plus = np.where(atr > 0, dm_plus_smooth / atr * 100, 0)
    di_minus = np.where(atr > 0, dm_minus_smooth / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = smooth_series(dx, 14)
    
    # Calculate daily Donchian channels (20-period)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    
    # Upper and lower bands
    upper_20 = np.full_like(daily_high, np.nan)
    lower_20 = np.full_like(daily_low, np.nan)
    
    for i in range(len(daily_high)):
        if i >= 19:
            upper_20[i] = np.max(daily_high[i-19:i+1])
            lower_20[i] = np.min(daily_low[i-19:i+1])
    
    # Calculate daily Donchian channels (10-period for exit)
    upper_10 = np.full_like(daily_high, np.nan)
    lower_10 = np.full_like(daily_low, np.nan)
    
    for i in range(len(daily_high)):
        if i >= 9:
            upper_10[i] = np.max(daily_high[i-9:i+1])
            lower_10[i] = np.min(daily_low[i-9:i+1])
    
    # Calculate daily average volume for volume filter
    daily_volume = df_daily['volume'].values
    vol_ma_20 = smooth_series(daily_volume, 20)
    
    # Align weekly ADX to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    
    # Align daily Donchian levels to daily timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_daily, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_daily, lower_20)
    upper_10_aligned = align_htf_to_ltf(prices, df_daily, upper_10)
    lower_10_aligned = align_htf_to_ltf(prices, df_daily, lower_10)
    
    # Align daily volume MA to daily timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(upper_10_aligned[i]) or np.isnan(lower_10_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current daily volume > 1.5x 20-day SMA
        vol_filter = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for breakout with volume confirmation and strong trend (ADX > 25)
            # Long: price breaks above 20-day high + ADX > 25 + volume spike
            if close[i] > upper_20_aligned[i] and adx_aligned[i] > 25:
                if vol_filter:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below 20-day low + ADX > 25 + volume spike
            elif close[i] < lower_20_aligned[i] and adx_aligned[i] > 25:
                if vol_filter:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below 10-day low or ADX falls below 20
            if close[i] < lower_10_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 10-day high or ADX falls below 20
            if close[i] > upper_10_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals