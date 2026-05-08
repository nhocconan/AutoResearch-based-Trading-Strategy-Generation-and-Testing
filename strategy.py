#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ADX filter
# Uses 4h price breakout above/below 20-period Donchian channel, confirmed by daily volume > 1.5x 20-day EMA and ADX > 25
# Exits when price returns to the 20-day EMA or ADX weakens below 20
# Designed to capture strong trending moves while avoiding choppy markets
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag

name = "4h_Donchian_Breakout_1dVolume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for volume and ADX
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate daily volume EMA (20-period)
    vol_ema_20 = pd.Series(df_daily['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate daily ADX (14-period)
    # True Range
    tr1 = df_daily['high'].values[1:] - df_daily['low'].values[1:]
    tr2 = np.abs(df_daily['high'].values[1:] - df_daily['close'].values[:-1])
    tr3 = np.abs(df_daily['low'].values[1:] - df_daily['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((df_daily['high'].values[1:] - df_daily['high'].values[:-1]) > (df_daily['low'].values[:-1] - df_daily['low'].values[1:]), 
                       np.maximum(df_daily['high'].values[1:] - df_daily['high'].values[:-1], 0), 0)
    dm_minus = np.where((df_daily['low'].values[:-1] - df_daily['low'].values[1:]) > (df_daily['high'].values[1:] - df_daily['high'].values[:-1]), 
                        np.maximum(df_daily['low'].values[:-1] - df_daily['low'].values[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period]) / period
        for i in range(period, len(data)):
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
    
    # Align daily indicators to 4h timeframe
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ema_20)
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Pre-compute 4h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h 20-period EMA for exit
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)  # warmup
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if np.isnan(vol_ema_20_aligned[i]) or np.isnan(adx_aligned[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current daily volume > 1.5x 20-day EMA
        # Find the most recent completed daily bar
        idx_daily = 0
        while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
            idx_daily += 1
        idx_daily -= 1  # last completed daily bar
        
        if idx_daily < 0:
            vol_filter = False
        else:
            vol_daily_current = df_daily.iloc[idx_daily]['volume']
            vol_filter = vol_daily_current > 1.5 * vol_ema_20_aligned[i]
        
        # ADX filter: > 25 indicates strong trending market
        adx_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Look for breakout entry with volume and ADX confirmation
            if close[i] > donchian_high[i-1] and vol_filter and adx_filter:
                signals[i] = 0.25
                position = 1
            elif close[i] < donchian_low[i-1] and vol_filter and adx_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 20 EMA or ADX weakens
            if close[i] <= ema_20[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 20 EMA or ADX weakens
            if close[i] >= ema_20[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals