#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for context (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily Williams Alligator components
    # Jaw (blue): 13-period SMA, shifted 8 bars forward
    # Teeth (red): 8-period SMA, shifted 5 bars forward
    # Lips (green): 5-period SMA, shifted 3 bars forward
    close_1d_series = pd.Series(close_1d)
    jaw_1d = close_1d_series.rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_1d = close_1d_series.rolling(window=8, min_periods=8).mean().shift(5).values
    lips_1d = close_1d_series.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 6h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate daily ADX for trend strength
    # Calculate True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    up_move = np.diff(high_1d, prepend=np.nan)
    down_move = -np.diff(low_1d, prepend=np.nan)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm[0] = minus_dm[0] = 0
    
    # Smooth TR and DM with Wilder's smoothing (using EMA as approximation)
    tr_period = 14
    tr_smooth = np.full(len(tr), np.nan)
    plus_dm_smooth = np.full(len(tr), np.nan)
    minus_dm_smooth = np.full(len(tr), np.nan)
    
    # Initial values
    if len(tr) >= tr_period:
        tr_smooth[tr_period-1] = np.nansum(tr[1:tr_period+1])
        plus_dm_smooth[tr_period-1] = np.nansum(plus_dm[1:tr_period+1])
        minus_dm_smooth[tr_period-1] = np.nansum(minus_dm[1:tr_period+1])
    
    # Wilder smoothing
    for i in range(tr_period, len(tr)):
        if not np.isnan(tr_smooth[i-1]):
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / tr_period) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / tr_period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / tr_period) + minus_dm[i]
    
    # Calculate DI and DX
    plus_di = np.full(len(tr), np.nan)
    minus_di = np.full(len(tr), np.nan)
    dx = np.full(len(tr), np.nan)
    
    for i in range(tr_period, len(tr)):
        if not np.isnan(tr_smooth[i]) and tr_smooth[i] != 0:
            plus_di[i] = 100 * (plus_dm_smooth[i] / tr_smooth[i])
            minus_di[i] = 100 * (minus_dm_smooth[i] / tr_smooth[i])
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # Calculate ADX (smoothed DX)
    adx_period = 14
    adx = np.full(len(tr), np.nan)
    
    if len(dx) >= 2 * adx_period:
        # Initial ADX value
        valid_dx = dx[adx_period:2*adx_period]
        if not np.all(np.isnan(valid_dx)):
            adx[2*adx_period-1] = np.nanmean(valid_dx)
        
        # Smooth ADX
        for i in range(2*adx_period, len(tr)):
            if not np.isnan(adx[i-1]) and not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator conditions: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_uptrend = lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i]
        alligator_downtrend = lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i]
        
        # ADX filter: trend strength > 25
        strong_trend = adx_aligned[i] > 25
        
        # Entry conditions
        long_entry = alligator_uptrend and strong_trend
        short_entry = alligator_downtrend and strong_trend
        
        # Exit conditions: trend weakens or Alligator reverses
        long_exit = not (alligator_uptrend and strong_trend)
        short_exit = not (alligator_downtrend and strong_trend)
        
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

name = "6h_1d_alligator_adx_trend_v1"
timeframe = "6h"
leverage = 1.0