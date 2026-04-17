#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Donchian breakout with 12h ADX trend filter and volume confirmation.
# Uses Donchian(20) breakouts for momentum, 12h ADX > 25 to filter for trending markets,
# and 6s volume > 1.5x 20-period average to avoid false breakouts.
# Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data for ADX trend filter ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = high_12h[1:] - low_12h[1:]
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
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        alpha = 1.0 / period
        smoothed = np.full_like(data, np.nan)
        smoothed[period-1] = np.nanmean(data[period-1:2*period-1])  # seed with simple average
        for i in range(period, len(data)):
            if not np.isnan(smoothed[i-1]):
                smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    tr_smooth = wilders_smooth(tr, 14)
    plus_dm_smooth = wilders_smooth(plus_dm, 14)
    minus_dm_smooth = wilders_smooth(minus_dm, 14)
    
    # DI and DX
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # === 6s Donchian channels ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 6s volume filter ===
    vol_avg20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # warmup for 12h indicators
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(vol_avg20[i])):
            signals[i] = 0.0
            continue
        
        vol_filter = volume[i] > 1.5 * vol_avg20[i]
        adx_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long: breakout above Donchian high + ADX trend + volume
            if high[i] > highest_high[i] and adx_filter and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: breakdown below Donchian low + ADX trend + volume
            if low[i] < lowest_low[i] and adx_filter and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long on breakdown below Donchian low or trend weakness
            if low[i] < lowest_low[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on breakout above Donchian high or trend weakness
            if high[i] > highest_high[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6s_Donchian_12hADX_VolumeFilter"
timeframe = "6h"
leverage = 1.0