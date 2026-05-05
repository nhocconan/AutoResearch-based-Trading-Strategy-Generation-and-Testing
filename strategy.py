#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w ADX(14) trend filter
# Long when price breaks above 4h Donchian(20) high AND 1d volume > 2x 20-period average AND 1w ADX > 25
# Short when price breaks below 4h Donchian(20) low AND 1d volume > 2x 20-period average AND 1w ADX > 25
# Exit when price crosses 4h Donchian(20) midpoint (mean reversion) OR 1w ADX < 20 (trend weakening)
# Uses 4h primary timeframe with 1d HTF for volume confirmation and 1w HTF for ADX trend filter
# Donchian channels provide clear breakout zones based on recent price extremes
# ADX filter ensures we only trade in trending markets, reducing whipsaw in ranges
# Volume confirmation filters low-momentum breakouts
# Discrete sizing (0.30) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Donchian20_Breakout_1dVolumeSpike_1wADX_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Wilder's smoothing function
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[1:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
            else:
                result[i] = np.nan
        return result
    
    tr_smooth = wilder_smooth(tr, 14)
    plus_dm_smooth = wilder_smooth(plus_dm, 14)
    minus_dm_smooth = wilder_smooth(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
    minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 1d volume 20-period average for volume spike filter
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_ma_20)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate 4h Donchian(20) channels
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1w_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND volume spike AND ADX > 25
            if (close[i] > donchian_high[i] and 
                volume_spike_1d_aligned[i] and 
                adx_1w_aligned[i] > 25):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Donchian low AND volume spike AND ADX > 25
            elif (close[i] < donchian_low[i] and 
                  volume_spike_1d_aligned[i] and 
                  adx_1w_aligned[i] > 25):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint (mean reversion) OR ADX < 20 (trend weakening)
            if close[i] < donchian_mid[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint (mean reversion) OR ADX < 20 (trend weakening)
            if close[i] > donchian_mid[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals