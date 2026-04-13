#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d volume spike and ADX regime filter
    # Long: price breaks above upper Donchian AND ADX(14) > 20 AND volume > 2x 20-period avg
    # Short: price breaks below lower Donchian AND ADX(14) > 20 AND volume > 2x 20-period avg
    # Exit: price returns to middle of Donchian channel OR opposite breakout
    # Using 4h timeframe for optimal trade frequency (target 19-50/year), Donchian for structure,
    # ADX to filter weak trends, volume spike to confirm institutional interest.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX and volume regime filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ADX(14) for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Wilder's smoothing function
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: smoothed = (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # DI+ and DI-
    plus_di14 = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di14 = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align daily ADX to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 4h Donchian channels (20-period)
    upper_donchian = np.full(n, np.nan)
    lower_donchian = np.full(n, np.nan)
    middle_donchian = np.full(n, np.nan)
    
    for i in range(20, n):
        upper_donchian[i] = np.max(high[i-20:i])
        lower_donchian[i] = np.min(low[i-20:i])
        middle_donchian[i] = (upper_donchian[i] + lower_donchian[i]) / 2
    
    # Calculate daily volume spike filter (>2x 20-period average)
    vol_ma_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_1d[i] = np.mean(df_1d['volume'].values[i-20:i])
    
    volume_spike_1d = np.full(len(df_1d), False)
    for i in range(20, len(df_1d)):
        if not np.isnan(vol_ma_1d[i]):
            volume_spike_1d[i] = df_1d['volume'].values[i] > (2.0 * vol_ma_1d[i])
    
    # Align daily volume spike to 4h
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(upper_donchian[i]) or 
            np.isnan(lower_donchian[i]) or np.isnan(middle_donchian[i]) or
            np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        trending_market = adx_1d_aligned[i] > 20
        vol_spike = bool(volume_spike_1d_aligned[i])
        
        # Donchian breakout conditions
        breakout_upper = close[i] > upper_donchian[i]
        breakout_lower = close[i] < lower_donchian[i]
        
        # Exit conditions: return to middle or opposite breakout
        return_to_middle = (position == 1 and close[i] < middle_donchian[i]) or \
                          (position == -1 and close[i] > middle_donchian[i])
        opposite_breakout = (position == 1 and breakout_lower) or \
                           (position == -1 and breakout_upper)
        
        # Entry logic: Donchian breakout + trending market + volume confirmation
        long_entry = breakout_upper and trending_market and vol_spike
        short_entry = breakout_lower and trending_market and vol_spike
        
        # Exit logic: return to middle or opposite breakout
        long_exit = return_to_middle or opposite_breakout
        short_exit = return_to_middle or opposite_breakout
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
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

name = "4h_1d_donchian_breakout_adx_volume_v2"
timeframe = "4h"
leverage = 1.0