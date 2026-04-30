#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h ADX25 trend filter and volume confirmation
# Uses Donchian channel breakouts for structure, traded only in direction of 12h ADX>25 (trending market)
# Volume spike (2.0x 20-period average) confirms institutional participation
# Works in bull markets via buying breakouts in uptrends and bear markets via selling breakdowns in downtrends
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Donchian20_Breakout_12hADX25_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop (MTF Rule #1)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:
        return np.zeros(n)
    
    # Calculate 12h ADX25
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(values[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_12h = wilders_smoothing(tr, 25)
    plus_di_12h = 100 * wilders_smoothing(plus_dm, 25) / atr_12h
    minus_di_12h = 100 * wilders_smoothing(minus_dm, 25) / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = wilders_smoothing(dx_12h, 25)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian calculation
    
    for i in range(start_idx, n):
        # Donchian channel: 20-period high/low
        donchian_high = np.max(high[i-20:i])
        donchian_low = np.min(low[i-20:i])
        
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_adx_12h = adx_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trending market (ADX > 25)
            if volume_spike and not np.isnan(curr_adx_12h) and curr_adx_12h > 25:
                # Bullish entry: price breaks above Donchian high
                if curr_close > donchian_high:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below Donchian low
                elif curr_close < donchian_low:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below Donchian low or ADX < 20 (trend weak)
            if curr_close < donchian_low or (not np.isnan(curr_adx_12h) and curr_adx_12h < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above Donchian high or ADX < 20 (trend weak)
            if curr_close > donchian_high or (not np.isnan(curr_adx_12h) and curr_adx_12h < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals