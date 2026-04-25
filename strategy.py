#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_RegimeFilter
Hypothesis: 4-hour Camarilla R1/S1 breakout with 1-day EMA34 trend filter and ADX regime filter.
Targets 20-30 trades/year by requiring: 1) price breaks daily R1/S1 levels, 2) aligned with 1d EMA34 trend,
3) ADX > 25 (trending market) for breakout entries. Uses 4h timeframe to balance trade frequency and capture
significant moves. The ADX filter avoids false breakouts in ranging markets and improves performance in both
bull and bear markets by only trading when trend strength is sufficient.
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
    
    # 1d data for Camarilla pivots and EMA34 (loaded ONCE)
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
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ADX for regime filter (trend strength)
    # TR = max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr_1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                       np.maximum(np.abs(df_1d['high'].values - df_1d['close'].shift(1).values),
                                  np.abs(df_1d['low'].values - df_1d['close'].shift(1).values)))
    # +DM = high - high_prev if high - high_prev > low_prev - low else 0
    dm_plus_1d = np.where((df_1d['high'].values - df_1d['high'].shift(1).values) > 
                          (df_1d['low'].shift(1).values - df_1d['low'].values),
                          np.maximum(df_1d['high'].values - df_1d['high'].shift(1).values, 0), 0)
    # -DM = low_prev - low if low_prev - low > high - high_prev else 0
    dm_minus_1d = np.where((df_1d['low'].shift(1).values - df_1d['low'].values) > 
                           (df_1d['high'].values - df_1d['high'].shift(1).values),
                           np.maximum(df_1d['low'].shift(1).values - df_1d['low'].values, 0), 0)
    # Smoothed TR, +DM, -DM
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    dm_plus_smoothed = pd.Series(dm_plus_1d).rolling(window=14, min_periods=14).mean().values
    dm_minus_smoothed = pd.Series(dm_minus_1d).rolling(window=14, min_periods=14).mean().values
    # DI+ and DI-
    di_plus_1d = 100 * dm_plus_smoothed / atr_1d
    di_minus_1d = 100 * dm_minus_smoothed / atr_1d
    # DX = |DI+ - DI-| / (DI+ + DI-) * 100
    dx_1d = 100 * np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d + 1e-10)
    # ADX = smoothed DX
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d previous data (1) + 1d EMA34 (34) + 1d ADX (14+14)
    start_idx = 34 + 14 + 14 + 1  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Regime filter: ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Look for entry signals with trend alignment and regime filter
            # Long breakout: price breaks above R1 with uptrend and trending market
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and trending
            # Short breakout: price breaks below S1 with downtrend and trending market
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and trending
            
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
            # Long position: exit if trend breaks or ADX weakens (range market)
            if not uptrend or adx_1d_aligned[i] < 20:  # Exit on trend change or ranging
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if trend breaks or ADX weakens (range market)
            if not downtrend or adx_1d_aligned[i] < 20:  # Exit on trend change or ranging
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_RegimeFilter"
timeframe = "4h"
leverage = 1.0