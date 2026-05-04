#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Long when price breaks above Donchian(20) high AND weekly pivot shows bullish bias (price > weekly PP) AND volume > 1.5x 20 EMA
# Short when price breaks below Donchian(20) low AND weekly pivot shows bearish bias (price < weekly PP) AND volume > 1.5x 20 EMA
# Uses 6h for structure, 1d for weekly pivot calculation, volume for confirmation
# Discrete sizing (0.25) to balance return and drawdown. Target: 12-37 trades/year.
# Weekly pivot provides structural bias that works in both bull (buy dips) and bear (sell rallies) markets.

name = "6h_Donchian20_WeeklyPivot_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation (based on prior week OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly OHLC from daily data
    # Group by week: Friday close as weekly close, etc.
    # We'll use prior week's OHLC for current week's pivot
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate weekly high, low, close using 5-day aggregation (approximation)
    # For simplicity, use prior 5-day period as proxy for weekly
    def rolling_apply(arr, window, func):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = func(arr[i-window+1:i+1])
        return result
    
    # Weekly high = max of prior 5 days high
    weekly_high = rolling_apply(high_1d, 5, np.max)
    # Weekly low = min of prior 5 days low
    weekly_low = rolling_apply(low_1d, 5, np.min)
    # Weekly close = last close of prior 5 days
    weekly_close = rolling_apply(close_1d, 5, lambda x: x[-1])
    # Weekly open = first open of prior 5 days
    weekly_open = rolling_apply(open_1d, 5, lambda x: x[0])
    
    # Calculate weekly pivot points
    # PP = (H + L + C) / 3
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # R1 = (2 * PP) - L
    weekly_r1 = 2 * weekly_pp - weekly_low
    # S1 = (2 * PP) - H
    weekly_s1 = 2 * weekly_pp - weekly_high
    # R2 = PP + (H - L)
    weekly_r2 = weekly_pp + (weekly_high - weekly_low)
    # S2 = PP - (H - L)
    weekly_s2 = weekly_pp - (weekly_high - weekly_low)
    # R3 = H + 2*(PP - L)
    weekly_r3 = weekly_high + 2 * (weekly_pp - weekly_low)
    # S3 = L - 2*(H - PP)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pp)
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    
    # Get 6h data for Donchian channel - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate 6h Donchian(20) - highest high and lowest low of prior 20 periods
    def donchian_channels(high_arr, low_arr, window):
        dch_high = np.full_like(high_arr, np.nan)
        dch_low = np.full_like(low_arr, np.nan)
        for i in range(window-1, len(high_arr)):
            dch_high[i] = np.max(high_arr[i-window+1:i+1])
            dch_low[i] = np.min(low_arr[i-window+1:i+1])
        return dch_high, dch_low
    
    donch_high, donch_low = donchian_channels(high_6h, low_6h, 20)
    
    # Align Donchian levels to 6h timeframe (already aligned since same TF)
    donch_high_aligned = donch_high  # Already 6h aligned
    donch_low_aligned = donch_low    # Already 6h aligned
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND bullish weekly bias (price > PP) AND volume spike
            if (close[i] > donch_high_aligned[i] and 
                close[i] > pp_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND bearish weekly bias (price < PP) AND volume spike
            elif (close[i] < donch_low_aligned[i] and 
                  close[i] < pp_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR weekly bias turns bearish
            if (close[i] < donch_low_aligned[i] or 
                close[i] < pp_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high OR weekly bias turns bullish
            if (close[i] > donch_high_aligned[i] or 
                close[i] > pp_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals