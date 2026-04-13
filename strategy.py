#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + ADX regime filter
    # Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
    # Long when Bull Power > 0 and ADX > 25 (trending up)
    # Short when Bear Power > 0 and ADX > 25 (trending down)
    # Uses 1d HTF for EMA(13) and ADX to reduce noise and avoid whipsaw
    # Works in bull (catch trends) and bear (catch downtrends) with regime filter
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h data for primary timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 1d data for HTF indicators (EMA and ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(13) for Elder Ray
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d ADX(14)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # first value is simple average
        result[period-1] = np.nanmean(data[:period])
        # subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr1d = wilder_smooth(tr, 14)
    plus_dm1d = wilder_smooth(plus_dm, 14)
    minus_dm1d = wilder_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di1d = np.where(atr1d != 0, (plus_dm1d / atr1d) * 100, 0)
    minus_di1d = np.where(atr1d != 0, (minus_dm1d / atr1d) * 100, 0)
    
    dx1d = np.where((plus_di1d + minus_di1d) != 0, 
                    np.abs((plus_di1d - minus_di1d) / (plus_di1d + minus_di1d)) * 100, 0)
    adx1d = wilder_smooth(dx1d, 14)
    
    # Align 1d indicators to 6h timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    adx1d_aligned = align_htf_to_ltf(prices, df_1d, adx1d)
    
    # Calculate Elder Ray on 6h using 1d EMA(13)
    bull_power = high_6h - ema13_1d_aligned
    bear_power = ema13_1d_aligned - low_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(1, n):  # start from 1 to access previous bar
        # Skip if data not ready
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(adx1d_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Regime: ADX > 25 indicates trending market
        is_trending = adx1d_aligned[i] > 25
        
        # Entry conditions
        long_entry = is_trending and (bull_power[i] > 0) and (position != 1)
        short_entry = is_trending and (bear_power[i] > 0) and (position != -1)
        
        # Exit conditions: reverse signal or ADX weakens
        exit_long = (position == 1 and (bull_power[i] <= 0 or adx1d_aligned[i] < 20))
        exit_short = (position == -1 and (bear_power[i] <= 0 or adx1d_aligned[i] < 20))
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0