#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h EMA crossover with 1d ADX trend filter and 1d Donchian breakout
    # Long: 6h EMA(9) > EMA(21) AND 1d ADX > 25 AND price > 1d Donchian(20) upper
    # Short: 6h EMA(9) < EMA(21) AND 1d ADX > 25 AND price < 1d Donchian(20) lower
    # Exit: EMA crossover reversal OR ADX < 20 (trend weak)
    # Using 6h for entries, 1d for trend/structure filters to avoid whipsaw.
    # Discrete position sizing (0.25) to minimize fee churn.
    # Target: 50-150 total trades over 4 years.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for ADX and Donchian filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Wilder's smoothing (14-period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[1:period])
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
    
    # Calculate 1d Donchian(20) channels
    donchian_len = 20
    upper = np.full(len(high_1d), np.nan)
    lower = np.full(len(low_1d), np.nan)
    for i in range(donchian_len-1, len(high_1d)):
        upper[i] = np.max(high_1d[i-donchian_len+1:i+1])
        lower[i] = np.min(low_1d[i-donchian_len+1:i+1])
    
    # Align 1d indicators to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    # Calculate 6h EMA(9) and EMA(21)
    ema_fast = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # EMA crossover signals
    ema_cross = np.zeros(n)
    for i in range(1, n):
        if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]:
            ema_cross[i] = 1   # bullish crossover
        elif ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]:
            ema_cross[i] = -1  # bearish crossover
        else:
            ema_cross[i] = ema_cross[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(ema_cross[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 = strong trend
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20  # exit when trend weakens
        
        # Breakout filters
        breakout_up = close[i] > upper_aligned[i]
        breakout_down = close[i] < lower_aligned[i]
        
        # Entry logic
        long_entry = (ema_cross[i] == 1) and strong_trend and breakout_up
        short_entry = (ema_cross[i] == -1) and strong_trend and breakout_down
        
        # Exit logic
        long_exit = (ema_cross[i] == -1) or weak_trend
        short_exit = (ema_cross[i] == 1) or weak_trend
        
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

name = "6h_1d_ema_adx_donchian_v1"
timeframe = "6h"
leverage = 1.0