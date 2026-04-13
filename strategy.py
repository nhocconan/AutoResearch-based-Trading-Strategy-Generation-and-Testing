#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h ADX regime filter
    # Long: Bull Power > 0 AND Bear Power < 0 AND 12h ADX > 25 (trending) AND 12h +DI > -DI
    # Short: Bear Power < 0 AND Bull Power > 0 AND 12h ADX > 25 AND 12h -DI > +DI
    # Exit: Elder Power divergence or ADX < 20 (range) -> flatten
    # Using 12h for regime (ADX) and 6h for Elder Ray calculation
    # Discrete position sizing (0.25) to minimize fee drag
    # Target: 50-150 total trades over 4 years (~12-37/year)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_time = prices['open_time'].values
    
    # Get 12h data for ADX regime filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX (14-period)
    period = 14
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value: simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        # Wilder smoothing: today = prev * (1-1/period) + current * (1/period)
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
        return result
    
    atr = wilders_smoothing(tr, period)
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    
    # +DI and -DI
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smoothing(dx, period)
    
    # Align 12h ADX, +DI, -DI to 6h
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    plus_di_12h_aligned = align_htf_to_ltf(prices, df_12h, plus_di)
    minus_di_12h_aligned = align_htf_to_ltf(prices, df_12h, minus_di)
    
    # 6h Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # warmup for EMA13 and Wilder smoothing
        # Skip if data not ready
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(plus_di_12h_aligned[i]) or 
            np.isnan(minus_di_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 12h ADX > 25 (trending market)
        regime_trending = adx_12h_aligned[i] > 25
        
        # Elder Ray signals
        long_signal = bull_power[i] > 0 and bear_power[i] < 0
        short_signal = bear_power[i] < 0 and bull_power[i] > 0  # same as above, kept for clarity
        
        # Direction from 12h DI crossover
        long_dir = plus_di_12h_aligned[i] > minus_di_12h_aligned[i]
        short_dir = minus_di_12h_aligned[i] > plus_di_12h_aligned[i]
        
        # Entry logic
        long_entry = long_signal and regime_trending and long_dir
        short_entry = short_signal and regime_trending and short_dir
        
        # Exit logic: ADX < 20 (range) or Elder Power divergence
        long_exit = (adx_12h_aligned[i] < 20) or (bull_power[i] <= 0)
        short_exit = (adx_12h_aligned[i] < 20) or (bear_power[i] >= 0)
        
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

name = "6h_12h_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0