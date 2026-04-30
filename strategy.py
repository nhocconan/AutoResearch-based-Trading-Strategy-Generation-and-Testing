#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Donchian(20) breakout + 1d ADX trend filter + volume confirmation
# Donchian channels provide clear breakout levels; 1d ADX > 25 filters for strong trending regimes.
# Volume spike (1.8x 20-period average) confirms institutional participation.
# Uses 6h timeframe for signal generation, 1d for trend regime. Discrete sizing 0.25 to balance return and risk.
# Target: 80-120 total trades over 4 years (20-30/year) to minimize fee drag while capturing trends.

name = "6h_Donchian20_Breakout_1dADX25_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) with proper smoothing
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with indices
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period])  # skip index 0 (nan)
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h Donchian channels (20-period) using prior completed bar
    donchian_window = 20
    high_max = np.full_like(high, np.nan)
    low_min = np.full_like(low, np.nan)
    
    for i in range(donchian_window, len(high)):
        high_max[i] = np.max(high[i-donchian_window:i])
        low_min[i] = np.min(low[i-donchian_window:i])
    
    # Align Donchian levels (already 6h, no need for MTF alignment)
    # But we need to ensure we use prior bar's levels to avoid look-ahead
    donchian_high = np.roll(high_max, 1)  # shift right to use prior bar
    donchian_low = np.roll(low_min, 1)
    donchian_high[0] = np.nan  # first value invalid after roll
    donchian_low[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, donchian_window)  # warmup
    
    for i in range(start_idx, n):
        # Skip if ADX not available or weak trend
        if np.isnan(adx_aligned[i]) or adx_aligned[i] < 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        
        # Volume confirmation: volume > 1.8x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[max(0, i-20):i])
            volume_spike = curr_volume > (1.8 * vol_ma_20)
        else:
            volume_spike = False
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: price breaks above Donchian high
                if not np.isnan(curr_donchian_high) and curr_close > curr_donchian_high:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below Donchian low
                elif not np.isnan(curr_donchian_low) and curr_close < curr_donchian_low:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below Donchian low or ADX weakens
            if curr_close < curr_donchian_low or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above Donchian high or ADX weakens
            if curr_close > curr_donchian_high or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals