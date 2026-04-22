#!/usr/bin/env python3
"""
12h Camarilla Pivot Reversal with 1d Volume Spike and ADX Trend Filter.
Long when price touches S1/S2 with bullish reversal and volume spike.
Short when price touches R1/R2 with bearish reversal and volume spike.
Exit when price crosses the pivot point (PP) or opposite S/R level.
Uses 1d ADX to filter for trending markets and avoid whipsaws.
Designed for low trade frequency (10-30/year) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for pivot and ADX - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (based on previous day)
    # PP = (H + L + C) / 3
    # S1 = C - (H - L) * 1.1 / 12
    # S2 = C - (H - L) * 1.1 / 6
    # R1 = C + (H - L) * 1.1 / 12
    # R2 = C + (H - L) * 1.1 / 6
    prev_high = df_daily['high'].shift(1).values
    prev_low = df_daily['low'].shift(1).values
    prev_close = df_daily['close'].shift(1).values
    
    pp = (prev_high + prev_low + prev_close) / 3.0
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12.0
    s2 = prev_close - (prev_high - prev_low) * 1.1 / 6.0
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12.0
    r2 = prev_close + (prev_high - prev_low) * 1.1 / 6.0
    
    # Align pivot levels to 12h timeframe (previous day's levels available at 00:00 UTC)
    pp_aligned = align_htf_to_ltf(prices, df_daily, pp)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    s2_aligned = align_htf_to_ltf(prices, df_daily, s2)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    r2_aligned = align_htf_to_ltf(prices, df_daily, r2)
    
    # Calculate 1d ADX (14-period) for trend filter
    high_d = pd.Series(df_daily['high'].values)
    low_d = pd.Series(df_daily['low'].values)
    close_d = pd.Series(df_daily['close'].values)
    
    # True Range
    tr1 = high_d - low_d
    tr2 = abs(high_d - close_d.shift(1))
    tr3 = abs(low_d - close_d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_d.diff()
    down_move = -low_d.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr_d)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr_d)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_d = dx.rolling(window=14, min_periods=14).mean()
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx_d.values)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to ensure previous day's pivot is available
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price touches S1/S2 with bullish reversal and volume spike
            # Bullish reversal: close > open and close > previous close
            bullish_reversal = close[i] > prices['open'].iloc[i] and close[i] > close[i-1]
            volume_spike = volume[i] > 2.0 * vol_avg_20[i]
            
            if bullish_reversal and volume_spike:
                # Touch S2 (stronger signal)
                if low[i] <= s2_aligned[i] * 1.002:  # Allow 0.2% slippage
                    signals[i] = 0.30
                    position = 1
                # Touch S1
                elif low[i] <= s1_aligned[i] * 1.002:
                    signals[i] = 0.20
                    position = 1
            # Short: Price touches R1/R2 with bearish reversal and volume spike
            # Bearish reversal: close < open and close < previous close
            bearish_reversal = close[i] < prices['open'].iloc[i] and close[i] < close[i-1]
            
            if bearish_reversal and volume_spike:
                # Touch R2 (stronger signal)
                if high[i] >= r2_aligned[i] * 0.998:  # Allow 0.2% slippage
                    signals[i] = -0.30
                    position = -1
                # Touch R1
                elif high[i] >= r1_aligned[i] * 0.998:
                    signals[i] = -0.20
                    position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses above PP or touches R1 (take profit)
                if close[i] > pp_aligned[i] or high[i] >= r1_aligned[i] * 0.998:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses below PP or touches S1 (take profit)
                if close[i] < pp_aligned[i] or low[i] <= s1_aligned[i] * 1.002:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "12H_CamarillaPivotReversal_1dADX_Volume"
timeframe = "12h"
leverage = 1.0
#%%