#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1w ADX trend filter and 1w Donchian breakout
# ADX(14) > 25 identifies strong trends to avoid whipsaw in choppy markets
# Donchian(20) breakouts capture momentum in trending regimes
# Works in bull markets (catch uptrends) and bear markets (catch downtrends)
# Uses weekly timeframe for trend and breakout to reduce noise and overtrading
# Target: 15-30 trades per year, low frequency for minimal fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1w data ONCE for ADX and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w ADX (14 periods)
    adx_len = 14
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (using Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmooth(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan)
        alpha = 1.0 / period
        smoothed = np.full_like(data, np.nan)
        smoothed[period-1] = np.nansum(data[1:period])  # First smoothed value
        for i in range(period, len(data)):
            if not np.isnan(smoothed[i-1]) and not np.isnan(data[i]):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + data[i]
            else:
                smoothed[i] = np.nan
        return smoothed
    
    tr_smoothed = WilderSmooth(tr, adx_len)
    plus_dm_smoothed = WilderSmooth(plus_dm, adx_len)
    minus_dm_smoothed = WilderSmooth(minus_dm, adx_len)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = WilderSmooth(dx, adx_len)
    
    # Calculate 1w Donchian channels (20 periods)
    donch_len = 20
    donch_high = pd.Series(high_1w).rolling(window=donch_len, min_periods=donch_len).max().values
    donch_low = pd.Series(low_1w).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # Align ADX and Donchian channels to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # Need enough for ADX calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Breakout signals from 1w Donchian
        breakout_up = price > donch_high_aligned[i]
        breakout_down = price < donch_low_aligned[i]
        
        if position == 0:
            # Enter long: strong trend + upward breakout
            if strong_trend and breakout_up:
                position = 1
                signals[i] = position_size
            # Enter short: strong trend + downward breakout
            elif strong_trend and breakout_down:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low OR trend weakens (ADX < 20)
            if price < donch_low_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high OR trend weakens
            if price > donch_high_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1wADX_1wDonchian_TrendBreakout_v1"
timeframe = "1d"
leverage = 1.0