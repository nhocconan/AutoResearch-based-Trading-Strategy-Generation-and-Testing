#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above Donchian upper band, 1d ADX > 25, volume > 2.0x 20-bar average
# Short when price breaks below Donchian lower band, 1d ADX > 25, volume > 2.0x 20-bar average
# Uses Donchian channels for structure, 1d ADX for trend strength filter, volume for momentum confirmation
# Designed for low trade frequency (~19-50/year on 4h) to minimize fee drag
# Works in bull (breakouts with rising volume in strong uptrend) and bear (breakdowns with rising volume in strong downtrend)

name = "4h_Donchian20_Volume_1dADX25_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.full_like(tr, np.nan, dtype=float)
    plus_dm_smooth = np.full_like(plus_dm, np.nan, dtype=float)
    minus_dm_smooth = np.full_like(minus_dm, np.nan, dtype=float)
    
    # Initial values (simple average of first 'period' values)
    if len(tr) >= period + 1:
        atr[period] = np.nanmean(tr[1:period+1])
        plus_dm_smooth[period] = np.nanmean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.nanmean(minus_dm[1:period+1])
        
        # Wilder's smoothing for subsequent values
        for i in range(period + 1, len(tr)):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
            plus_dm_smooth[i] = alpha * plus_dm[i] + (1 - alpha) * plus_dm_smooth[i-1]
            minus_dm_smooth[i] = alpha * minus_dm[i] + (1 - alpha) * minus_dm_smooth[i-1]
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.full_like(dx, np.nan, dtype=float)
    
    # Initial ADX value (simple average of first 'period' DX values)
    if len(dx) >= 2 * period:
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        # Wilder's smoothing for subsequent ADX values
        for i in range(2*period, len(dx)):
            adx[i] = alpha * dx[i] + (1 - alpha) * adx[i-1]
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (2.0x 20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(2*14, 20) + 1  # ADX(14,14) + Donchian(20) + volume MA(20) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when ADX > 25 (strong trend)
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Donchian upper band, strong trend, volume spike
            if (close[i] > highest_high[i] and strong_trend and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price < Donchian lower band, strong trend, volume spike
            elif (close[i] < lowest_low[i] and strong_trend and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Donchian lower band or trend weakens (ADX < 20)
            if (close[i] < lowest_low[i] or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Donchian upper band or trend weakens (ADX < 20)
            if (close[i] > highest_high[i] or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals