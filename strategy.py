#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour CAMARILLA PIVOT LEVELS + DAILY VOLUME CONFIRMATION + WEEKLY ADX TREND FILTER
# Long when price breaks above R1 with volume confirmation and weekly ADX > 25
# Short when price breaks below S1 with volume confirmation and weekly ADX > 25
# Exit when price returns to pivot point or weekly ADX falls below 20
# Camarilla pivots provide strong intraday support/resistance levels
# Volume confirms breakout strength, ADX filters for trending conditions
# Targets 15-25 trades/year to minimize fee decay while capturing sustained moves

name = "12h_CamarillaPivot_Volume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data for pivot points and volume confirmation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
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
    
    # Calculate daily average volume for volume filter
    daily_volume = df_daily['volume'].values
    vol_ma_20 = smooth_series(daily_volume, 20)
    
    # Align weekly ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    
    # Align daily volume MA to 12h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Find the most recent completed daily bar for pivot calculation
        idx_daily = len(df_daily) - 1
        while idx_daily >= 0 and df_daily.iloc[idx_daily]['open_time'] > prices.iloc[i]['open_time']:
            idx_daily -= 1
        
        if idx_daily < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla pivot levels for the most recent completed daily bar
        daily_high = df_daily.iloc[idx_daily]['high']
        daily_low = df_daily.iloc[idx_daily]['low']
        daily_close = df_daily.iloc[idx_daily]['close']
        
        pivot = (daily_high + daily_low + daily_close) / 3
        range_val = daily_high - daily_low
        
        # Camarilla levels
        r1 = pivot + (range_val * 1.1 / 12)
        s1 = pivot - (range_val * 1.1 / 12)
        
        # Volume filter: current daily volume > 1.5x 20-day SMA
        vol_daily_current = df_daily.iloc[idx_daily]['volume']
        vol_filter = vol_daily_current > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for Camarilla breakout with volume confirmation and strong trend
            # Long: price breaks above R1
            if close[i] > r1 and adx_aligned[i] > 25:
                if vol_filter:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below S1
            elif close[i] < s1 and adx_aligned[i] > 25:
                if vol_filter:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price returns to pivot point or ADX falls below 20
            if close[i] <= pivot or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to pivot point or ADX falls below 20
            if close[i] >= pivot or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals