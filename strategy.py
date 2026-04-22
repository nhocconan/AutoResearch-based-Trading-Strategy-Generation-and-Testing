#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d ADX trend filter and volume confirmation
# This strategy trades breakouts of the 4h Donchian(20) channel with trend alignment from 1d ADX
# and volume confirmation to avoid false breakouts. Works in both bull and bear markets by
# following the trend direction on higher timeframe. Uses discrete position sizing (0.25) to
# balance return and minimize transaction costs.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) for trend strength
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Plus and Minus Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0]) * -1  # Invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smooth_series(vals, period):
        smoothed = np.zeros_like(vals)
        smoothed[period-1] = np.mean(vals[:period])
        for i in range(period, len(vals)):
            smoothed[i] = (smoothed[i-1] * (period-1) + vals[i]) / period
        return smoothed
    
    atr = smooth_series(tr, 14)
    plus_di = 100 * smooth_series(plus_dm, 14) / atr
    minus_di = 100 * smooth_series(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_series(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian channel (20-period) on 4h
    def donchian_channels(high_vals, low_vals, period):
        upper = np.zeros_like(high_vals)
        lower = np.zeros_like(low_vals)
        for i in range(len(high_vals)):
            if i < period - 1:
                upper[i] = np.nan
                lower[i] = np.nan
            else:
                upper[i] = np.max(high_vals[i-period+1:i+1])
                lower[i] = np.min(low_vals[i-period+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 19:
            vol_avg_20[i] = np.nan
        else:
            vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to ensure Donchian is ready
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian upper + ADX > 25 (trending) + volume spike
            if close[i] > donchian_upper[i] and adx_aligned[i] > 25 and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower + ADX > 25 (trending) + volume spike
            elif close[i] < donchian_lower[i] and adx_aligned[i] > 25 and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back into Donchian channel
            if position == 1:
                # Exit long: Price below Donchian lower (or upper for re-entry prevention)
                if close[i] < donchian_lower[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price above Donchian upper
                if close[i] > donchian_upper[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dADX25_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0