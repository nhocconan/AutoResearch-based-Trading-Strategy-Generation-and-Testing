#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_Filter
Hypothesis: 4-hour Camarilla R1/S1 breakout with 12-hour EMA50 trend filter. Targets 25-35 trades/year by requiring: 
1) price breaks daily R1/S1 levels, 2) aligned with 12h EMA50 trend, 3) avoids choppy markets with ADX < 20 filter.
Uses 4h timeframe to balance trade frequency and capture significant moves. The ADX filter avoids false breakouts 
in ranging markets and improves performance in both bull and bear markets by only trading in clear trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for Camarilla pivots (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels (R1 = C + 1.1*(HL/4), S1 = C - 1.1*(HL/4))
    R1 = prev_close + 1.1 * prev_range * (1.0/4.0)
    S1 = prev_close - 1.1 * prev_range * (1.0/4.0)
    
    # Align 1d levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 12h data for EMA50 trend filter and ADX (loaded ONCE)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h ADX for trend strength filter (ADX < 20 = weak trend/ranging, avoid)
    # Calculate True Range
    tr_12h = np.maximum(df_12h['high'].values - df_12h['low'].values,
                        np.maximum(np.abs(df_12h['high'].values - df_12h['close'].shift(1).values),
                                   np.abs(df_12h['low'].values - df_12h['close'].shift(1).values)))
    # Calculate +DM and -DM
    up_move = df_12h['high'].values - np.roll(df_12h['high'].values, 1)
    down_move = np.roll(df_12h['low'].values, 1) - df_12h['low'].values
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smooth TR, +DM, -DM over 14 periods
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr_12h
    minus_di = 100 * minus_dm_smooth / atr_12h
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    # Replace NaN/inf with 0
    adx = np.nan_to_num(adx, nan=0.0, posinf=0.0, neginf=0.0)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d previous data (1) + 12h EMA50 (50) + 12h ADX (14+14=28)
    start_idx = 50 + 28 + 1  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 12h EMA50 and ADX > 20 (strong trend)
        uptrend = curr_close > ema_50_12h_aligned[i]
        downtrend = curr_close < ema_50_12h_aligned[i]
        strong_trend = adx_aligned[i] > 20
        
        if position == 0:
            # Look for entry signals with trend alignment and strong trend filter
            # Long breakout: price breaks above R1 with uptrend and strong trend
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and strong_trend
            # Short breakout: price breaks below S1 with downtrend and strong trend
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and strong_trend
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit if price breaks below S1 or trend changes to downtrend
            if curr_close < S1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if price breaks above R1 or trend changes to uptrend
            if curr_close > R1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_Filter"
timeframe = "4h"
leverage = 1.0