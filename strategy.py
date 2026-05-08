#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price action with daily volume confirmation and 1d ADX trend filter
# Uses 4h price action for entry timing, daily volume for confirmation, and 1d ADX to filter weak trends.
# Works in bull/bear via breakout logic with volume and trend strength filters.
# Target: 50-150 total trades over 4 years with low turnover to minimize fee drag.

name = "4h_PriceAction_DailyVolume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume average and ADX
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily volume average (34-period EMA)
    vol_ema_34 = pd.Series(df_daily['volume'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily ADX (34-period)
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
    
    atr = wilders_smoothing(tr, 34)
    dm_plus_smooth = wilders_smoothing(dm_plus, 34)
    dm_minus_smooth = wilders_smoothing(dm_minus, 34)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 34)
    
    # Align daily indicators to 4h timeframe
    vol_ema_34_aligned = align_htf_to_ltf(prices, df_daily, vol_ema_34)
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if np.isnan(vol_ema_34_aligned[i]) or np.isnan(adx_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current daily volume > 1.5x 34-day EMA
        # Find the most recent completed daily bar
        idx_daily = len(df_daily) - 1
        while idx_daily >= 0 and df_daily.iloc[idx_daily]['open_time'] > prices.iloc[i]['open_time']:
            idx_daily -= 1
        
        if idx_daily < 0:
            vol_filter = False
        else:
            vol_daily_current = df_daily.iloc[idx_daily]['volume']
            vol_filter = vol_daily_current > 1.5 * vol_ema_34_aligned[i]
        
        # ADX filter: > 25 indicates strong trend
        adx_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Look for breakout entry with volume and ADX confirmation
            if close[i] > close[i-1] and close[i-1] > close[i-2] and vol_filter and adx_filter:
                signals[i] = 0.25
                position = 1
            elif close[i] < close[i-1] and close[i-1] < close[i-2] and vol_filter and adx_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks down or ADX weakens
            if close[i] < close[i-1] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks up or ADX weakens
            if close[i] > close[i-1] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals