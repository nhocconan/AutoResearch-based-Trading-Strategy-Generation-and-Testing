#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume spike confirmation.
- Primary timeframe: 4h for lower trade frequency and better signal quality.
- HTF: 1d ADX(14) > 25 for trending regime, ADX < 20 for ranging (avoid false breakouts).
- Volume: Current 4h volume > 1.8 * 20-period volume MA to capture institutional interest.
- Donchian: Upper/lower bands from 20-period high/low on 4h.
- Entry: Long when price breaks above Upper Band AND 1d ADX > 25 AND volume spike.
         Short when price breaks below Lower Band AND 1d ADX > 25 AND volume spike.
- Exit: Price reverts to 20-period EMA on 4h or loss of volume confirmation/ADX < 20.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe.
This strategy captures institutional breakouts in trending markets, filtered by daily trend strength
to avoid choppy false breakouts. Volume spikes confirm participation, and ADX ensures we only
trade when there's sufficient trend strength to justify the breakout.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.abs(high_1d[0] - low_1d[0])], tr])
    
    # Directional Movement
    up_move = np.diff(high_1d)
    down_move = -np.diff(low_1d)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    def wilders_smooth(data, period):
        result = np.zeros_like(data)
        result[:] = np.nan
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_14 = wilders_smooth(tr, 14)
    plus_dm_14 = wilders_smooth(plus_dm, 14)
    minus_dm_14 = wilders_smooth(minus_dm, 14)
    
    # DI values
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = np.full_like(dx, np.nan)
    valid_idx = ~np.isnan(dx)
    if np.sum(valid_idx) >= 14:
        # First ADX is average of first 14 DX values
        first_adx_idx = np.where(valid_idx)[0][13] if np.sum(valid_idx) >= 14 else -1
        if first_adx_idx != -1:
            adx[first_adx_idx] = np.nanmean(dx[valid_idx][:14])
            # Subsequent ADX: Wilder's smoothing of DX
            for i in range(first_adx_idx + 1, len(dx)):
                if not np.isnan(dx[i]):
                    adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Calculate 20-period 1d volume MA
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Donchian bands (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.nanmax(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.nanmin(arr[i-window+1:i+1])
        return result
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    
    # Calculate 4h 20-period EMA for exit
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).values
    
    # Align HTF indicators to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 4h volume > 1.8 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20, 20)  # ADX, Donchian, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for breakout signals with volume spike and strong trend
            if volume_spike[i] and adx_1d_aligned[i] > 25:
                # Bullish breakout: price > Upper Band
                if curr_high > donchian_high[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price < Lower Band
                elif curr_low < donchian_low[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to EMA20 OR loss of volume confirmation OR weak trend
            if (curr_close <= ema_20[i] or not volume_spike[i] or adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to EMA20 OR loss of volume confirmation OR weak trend
            if (curr_close >= ema_20[i] or not volume_spike[i] or adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dADX_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0