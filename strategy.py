#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian breakout with 1w ADX trend filter and volume confirmation
    # Long: Price breaks above Donchian(20) high AND 1w ADX > 25 (trending) AND volume > 1.5x avg
    # Short: Price breaks below Donchian(20) low AND 1w ADX > 25 (trending) AND volume > 1.5x avg
    # Exit: Price crosses Donchian midpoint OR volume dry-up
    # Using 12h timeframe for low trade frequency, Donchian for clear structure,
    # 1w ADX for trend regime filter (avoid choppy markets), volume for confirmation.
    # Discrete position sizing (0.30) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly ADX(14) for trend regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[1:period])
        # Subsequent values
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    dpi = wilders_smoothing(dm_plus, 14)
    dmi = wilders_smoothing(dm_minus, 14)
    
    # Avoid division by zero
    di_plus = 100 * dpi / atr
    di_minus = 100 * dmi / atr
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    adx = wilders_smoothing(dx, 14)
    
    # Align weekly ADX to 12h
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Donchian midpoint for exit
    midpoint = (highest_high + lowest_low) / 2
    
    # Get 12h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 = trending market
        trending = adx_1w_aligned[i] > 25
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Breakout conditions
        long_breakout = close[i] > highest_high[i]
        short_breakout = close[i] < lowest_low[i]
        
        # Exit conditions
        long_exit = close[i] < midpoint[i]
        short_exit = close[i] > midpoint[i]
        
        # Entry logic: Breakout + trend + volume
        long_entry = long_breakout and trending and vol_confirm
        short_entry = short_breakout and trending and vol_confirm
        
        # Exit logic: midpoint cross OR volume dry-up
        long_exit_cond = long_exit or not vol_confirm
        short_exit_cond = short_exit or not vol_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.30
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and long_exit_cond:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit_cond:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_donchian_adx_volume_v1"
timeframe = "12h"
leverage = 1.0