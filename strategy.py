#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX(14)>25 trend filter and volume spike (>1.7x 20 EMA volume)
# Uses Donchian channel from prior completed 4h bar for structure (20-period high/low)
# 1d ADX filter ensures we only trade in trending markets, reducing whipsaw in ranging conditions
# Volume confirmation ensures breakout has institutional participation
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# This focuses on strong breakouts in trending markets with volume confirmation, which historically works well on BTC/ETH

name = "4h_Donchian20_1dADX_VolumeSpike"
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
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX(14) trend filter from prior completed 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan  # First value has no prior close
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=np.nan)
    down_move = -np.diff(low_1d, prepend=np.nan)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smoothed = wilders_smoothing(tr, period)
    plus_dm_smoothed = wilders_smoothing(plus_dm, period)
    minus_dm_smoothed = wilders_smoothing(minus_dm, period)
    
    # DI+ and DI-
    plus_di = np.where(tr_smoothed != 0, (plus_dm_smoothed / tr_smoothed) * 100, 0)
    minus_di = np.where(tr_smoothed != 0, (minus_dm_smoothed / tr_smoothed) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    adx = wilders_smoothing(dx, period)
    
    # Shift ADX by 1 to use only prior completed 1d bar (no look-ahead)
    adx_shifted = np.roll(adx, 1)
    adx_shifted[0] = np.nan
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_shifted)
    
    # Get 4h data for Donchian channel calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need enough data for Donchian calculation
        return np.zeros(n)
    
    # Calculate Donchian channel (20-period) from prior completed 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate rolling max/min for Donchian channels
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
    
    donchian_high = rolling_max(high_4h, 20)
    donchian_low = rolling_min(low_4h, 20)
    
    # Shift by 1 to use only prior completed 4h bar (no look-ahead)
    donchian_high_shifted = np.roll(donchian_high, 1)
    donchian_low_shifted = np.roll(donchian_low, 1)
    donchian_high_shifted[0] = np.nan
    donchian_low_shifted[0] = np.nan
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_shifted)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + ADX > 25 (trending) + volume spike
            if close[i] > donchian_high_aligned[i] and adx_1d_aligned[i] > 25 and volume[i] > (1.7 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + ADX > 25 (trending) + volume spike
            elif close[i] < donchian_low_aligned[i] and adx_1d_aligned[i] > 25 and volume[i] > (1.7 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of Donchian channel OR ADX drops below 20 (trend weakening)
            midpoint = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if not np.isnan(midpoint) and (close[i] < midpoint or adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint of Donchian channel OR ADX drops below 20 (trend weakening)
            midpoint = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if not np.isnan(midpoint) and (close[i] > midpoint or adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals