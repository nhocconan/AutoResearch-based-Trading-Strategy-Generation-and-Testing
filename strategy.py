#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with daily volume confirmation and weekly ADX trend filter
# Long when price breaks above R3 level + daily volume > 1.5x 20-period SMA + weekly ADX > 25
# Short when price breaks below S3 level + daily volume > 1.5x 20-period SMA + weekly ADX > 25
# Exit when price crosses opposite S1/R1 level or weekly ADX falls below 20
# Uses Camarilla pivots for institutional levels, volume for confirmation, ADX for trend strength
# Targets 25-35 trades/year to minimize fee drag while capturing strong directional moves in both bull and bear markets

name = "4h_Camarilla_R3S3_Breakout_DailyVolume_WeeklyADX"
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
    
    # Get daily data for Camarilla pivots and volume
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
    
    # Calculate daily Camarilla pivot levels (R3, S3, R1, S1)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Pivot point
    pivot = (daily_high + daily_low + daily_close) / 3.0
    
    # Camarilla levels
    camarilla_range = daily_high - daily_low
    r3 = pivot + camarilla_range * 1.1 / 4.0
    s3 = pivot - camarilla_range * 1.1 / 4.0
    r1 = pivot + camarilla_range * 1.1 / 12.0
    s1 = pivot - camarilla_range * 1.1 / 12.0
    
    # Calculate daily average volume for volume filter
    daily_volume = df_daily['volume'].values
    vol_ma_20 = smooth_series(daily_volume, 20)
    
    # Align weekly ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    
    # Align daily Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    
    # Align daily volume MA to 4h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current daily volume > 1.5x 20-day SMA
        # Find the most recent completed daily bar
        idx_daily = len(df_daily) - 1
        while idx_daily >= 0 and df_daily.iloc[idx_daily]['open_time'] > prices.iloc[i]['open_time']:
            idx_daily -= 1
        vol_filter = False
        if idx_daily >= 0:
            vol_daily_current = df_daily.iloc[idx_daily]['volume']
            vol_filter = vol_daily_current > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for breakout with volume confirmation and strong trend (ADX > 25)
            # Long: price breaks above R3 level + ADX > 25 + volume spike
            if close[i] > r3_aligned[i] and adx_aligned[i] > 25:
                if vol_filter:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below S3 level + ADX > 25 + volume spike
            elif close[i] < s3_aligned[i] and adx_aligned[i] > 25:
                if vol_filter:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below S1 level or ADX falls below 20
            if close[i] < s1_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above R1 level or ADX falls below 20
            if close[i] > r1_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals