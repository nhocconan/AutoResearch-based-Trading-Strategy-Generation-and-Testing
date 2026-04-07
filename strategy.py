#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian(20) breakout with volume confirmation and ADX trend filter
# Hypothesis: Breakouts capture directional moves; volume confirms institutional participation; ADX filters choppy markets.
# Works in bull via upward breakouts, in bear via downward breakdowns. ADX prevents whipsaws in ranging markets.
# Target: 20-50 trades/year to minimize fee drag.
name = "4h_donchian20_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily 20-period volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate ADX(14) on daily timeframe for trend strength
    # True Range
    tr1 = df_1d['high'].values[1:] - df_1d['low'].values[1:]
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr_daily = np.concatenate([[np.max([df_1d['high'].values[0] - df_1d['low'].values[0], 
                                       np.abs(df_1d['high'].values[0] - df_1d['close'].values[0]), 
                                       np.abs(df_1d['low'].values[0] - df_1d['close'].values[0])])], 
                              np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = df_1d['high'].values[1:] - df_1d['high'].values[:-1]
    down_move = df_1d['low'].values[:-1] - df_1d['low'].values[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    atr_daily = np.zeros_like(tr_daily)
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    # Initial values
    atr_daily[0] = np.mean(tr_daily[:14]) if len(tr_daily) >= 14 else np.nan
    plus_dm_smooth[0] = np.mean(plus_dm[:14]) if len(plus_dm) >= 14 else np.nan
    minus_dm_smooth[0] = np.mean(minus_dm[:14]) if len(minus_dm) >= 14 else np.nan
    
    # Wilder smoothing
    for i in range(1, len(tr_daily)):
        atr_daily[i] = (atr_daily[i-1] * 13 + tr_daily[i]) / 14
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * 13 + plus_dm[i]) / 14
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * 13 + minus_dm[i]) / 14
    
    # Calculate DI+ and DI-
    plus_di = np.where(atr_daily != 0, (plus_dm_smooth / atr_daily) * 100, 0)
    minus_di = np.where(atr_daily != 0, (minus_dm_smooth / atr_daily) * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx_daily = np.zeros_like(dx)
    adx_daily[:14] = np.nan
    for i in range(14, len(dx)):
        adx_daily[i] = (adx_daily[i-1] * 13 + dx[i]) / 14
    
    # Align daily data to 4h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_daily)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band (20-period low)
            if close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band (20-period high)
            if close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above Donchian upper band + volume confirmation + trending
            if close[i] > highest_high[i] and vol_confirm and trending:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian lower band + volume confirmation + trending
            elif close[i] < lowest_low[i] and vol_confirm and trending:
                position = -1
                signals[i] = -0.25
    
    return signals