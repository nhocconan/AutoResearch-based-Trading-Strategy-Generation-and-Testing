#!/usr/bin/env python3
# 4h_1d_donchian_breakout_volume_v1
# Strategy: 4h Donchian(20) breakout with 1d volume confirmation and 1d ADX trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture strong directional moves. Volume confirmation
# ensures institutional participation. ADX > 25 filters choppy markets. Designed for
# 20-40 trades/year to minimize fee drag while capturing major trends in bull/bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def wilders_smoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    dm_plus14 = wilders_smoothing(dm_plus, 14)
    dm_minus14 = wilders_smoothing(dm_minus, 14)
    
    # DI and DX
    di_plus = np.where(tr14 != 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus14 / tr14, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    adx_1d = adx  # 14-period smoothed
    
    # 1d volume moving average
    vol_ma_20_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > (vol_ma_20_1d_aligned[i] * 1.5)
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous low
        
        # Entry logic: Donchian breakout + volume confirmation + trend filter
        if (breakout_up and volume_confirmed and trending and position != 1):
            position = 1
            signals[i] = 0.25
        elif (breakout_down and volume_confirmed and trending and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: Opposite Donchian breakout or loss of trend/volume conditions
        elif position == 1 and (breakout_down or not trending or not volume_confirmed):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (breakout_up or not trending or not volume_confirmed):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals