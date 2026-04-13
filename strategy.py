#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 12h ADX regime filter and volume confirmation
    # Long: price breaks above upper band AND 12h ADX > 20 (trending) AND volume > 1.3x avg
    # Short: price breaks below lower band AND 12h ADX > 20 (trending) AND volume > 1.3x avg
    # Exit: price touches middle band (mean reversion) OR opposite Donchian band
    # Using 6h timeframe for optimal trade frequency (target 12-37/year), Donchian for structure,
    # 12h ADX to filter ranging markets, and volume confirmation to avoid false breakouts.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h ADX(14) for trend strength filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Wilder's smoothing
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
    
    # Align 12h ADX to 6h
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate 6h Donchian channels (20-period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    # Get 6h volume for confirmation (>1.3x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(middle[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 20 indicates trending market
        trending_market = adx_12h_aligned[i] > 20
        
        # Donchian breakout conditions
        breakout_upper = close[i] > upper[i]
        breakout_lower = close[i] < lower[i]
        
        # Exit conditions: touch middle band or opposite band
        touch_middle = abs(close[i] - middle[i]) < (upper[i] - lower[i]) * 0.05  # within 5% of middle
        touch_opposite_lower = close[i] < lower[i] and position == 1  # Long exit on lower band retest
        touch_opposite_upper = close[i] > upper[i] and position == -1  # Short exit on upper band retest
        
        # Entry logic: Donchian breakout + trending market + volume confirmation
        long_entry = breakout_upper and trending_market and volume_spike[i]
        short_entry = breakout_lower and trending_market and volume_spike[i]
        
        # Exit logic: middle band touch or opposite band retest
        long_exit = touch_middle or touch_opposite_lower
        short_exit = touch_middle or touch_opposite_upper
        
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

name = "6h_12h_donchian_breakout_adx_volume_v1"
timeframe = "6h"
leverage = 1.0