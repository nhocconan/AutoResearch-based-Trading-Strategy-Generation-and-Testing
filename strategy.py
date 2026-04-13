#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian breakout with 1d ADX trend filter and volume confirmation
    # Long when price breaks above 20-period Donchian high + ADX > 25 (trending) + volume > 1.5x average
    # Short when price breaks below 20-period Donchian low + ADX > 25 (trending) + volume > 1.5x average
    # Uses discrete position sizing (0.25) to minimize fee churn
    # Target: 50-150 total trades over 4 years (~12-37/year)
    # Donchian provides objective breakout levels; ADX filters ranging markets; volume confirms strength
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for ADX, Donchian, and volume (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ADX (14-period) for trend filter
    # True Range
    tr1 = np.abs(np.roll(high_1d, 1) - low_1d)
    tr2 = np.abs(np.roll(low_1d, 1) - high_1d)
    tr3 = np.abs(high_1d - low_1d)
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: smoothed = (prev_smoothed * (period-1) + current) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Calculate 1d Donchian channels (20-period)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume average (20-period) with min_periods
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 12h (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5 * 20-period average
        volume_confirm = vol_1d_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_high_aligned[i] and volume_confirm and trending
        bearish_breakout = close[i] < donch_low_aligned[i] and volume_confirm and trending
        
        # Exit conditions: price returns to opposite Donchian level or opposite breakout
        long_exit = close[i] < donch_low_aligned[i] or bearish_breakout
        short_exit = close[i] > donch_high_aligned[i] or bullish_breakout
        
        if bullish_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_breakout and position != -1:
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

name = "12h_1d_donchian_breakout_adx_volume_v1"
timeframe = "12h"
leverage = 1.0