#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d ADX for trend strength and 1w Donchian breakouts for entry timing
# - Uses 1d HTF for ADX(14): ADX > 25 indicates strong trend (either bullish or bearish)
# - Uses 1w HTF for Donchian(20) breakouts: price breaks above/below 20-period weekly high/low
# - In strong trending markets (ADX > 25): trade breakout direction with the trend
# - Volume confirmation: current 6h volume > 1.5x 20-period average to filter low-quality breakouts
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets by capturing strong trend continuations

name = "6h_1d_1w_adx_donchian_v1"
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
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d ADX (14 periods)
    # True Range
    tr1 = pd.Series(high_1d).rolling(2).apply(lambda x: x[0] - x[1] if len(x) == 2 else 0, raw=True).shift(1).fillna(0).values
    tr2 = pd.Series(low_1d).rolling(2).apply(lambda x: x[0] - x[1] if len(x) == 2 else 0, raw=True).shift(1).fillna(0).values
    tr3 = abs(pd.Series(high_1d).diff(1).values)
    tr4 = abs(pd.Series(low_1d).diff(1).values)
    tr = np.maximum.reduce([tr1, tr2, tr3, tr4])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = pd.Series(high_1d).diff(1).values
    down_move = -pd.Series(low_1d).diff(1).values
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_period = wilder_smooth(tr, period)
    plus_dm_period = wilder_smooth(plus_dm, period)
    minus_dm_period = wilder_smooth(minus_dm, period)
    
    # Avoid division by zero
    plus_di = np.where(tr_period != 0, (plus_dm_period / tr_period) * 100, 0)
    minus_di = np.where(tr_period != 0, (minus_dm_period / tr_period) * 100, 0)
    dx = np.where((plus_di + minus_di) != 0, abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = wilder_smooth(dx, period)
    
    # Calculate 1w Donchian Channel (20 periods)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align all HTF data to 6h timeframe (wait for completed HTF bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Strong trend condition: ADX > 25
        strong_trend = adx_aligned[i] > 25
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions: trend weakens or opposite breakout
            if not strong_trend or breakout_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit conditions: trend weakens or opposite breakout
            if not strong_trend or breakout_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry logic: strong trend + volume-confirmed breakout
            if strong_trend and volume_confirmed:
                if breakout_up:
                    position = 1
                    signals[i] = position_size
                elif breakout_down:
                    position = -1
                    signals[i] = -position_size
    
    return signals