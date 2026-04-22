#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot breakout with 1d ADX filter and volume confirmation.
Long when price breaks above R1 with strong ADX trend and volume spike.
Short when price breaks below S1 with strong ADX trend and volume spike.
Exit when price crosses back below R1 or above S1, or ADX weakens.
Uses 1d ADX for trend strength filter to avoid whipsaws in ranging markets.
Designed for low trade frequency (20-50/year) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla_pivots(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    # Camarilla formula
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    multiplier = 1.1 / 12
    close_val = close
    S3 = close_val - (range_val * 1.1 * 0.5)
    S2 = close_val - (range_val * 1.1 * 0.382)
    S1 = close_val - (range_val * 1.1 * 0.25)
    PP = close_val
    R1 = close_val + (range_val * 1.1 * 0.25)
    R2 = close_val + (range_val * 1.1 * 0.382)
    R3 = close_val + (range_val * 1.1 * 0.5)
    return PP, S3, S2, S1, R1, R2, R3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Camarilla pivots and ADX - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivots
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    n_daily = len(high_d)
    
    PPs = np.zeros(n_daily)
    S1s = np.zeros(n_daily)
    R1s = np.zeros(n_daily)
    
    for i in range(n_daily):
        PP, S3, S2, S1, R1, R2, R3 = calculate_camarilla_pivots(high_d[i], low_d[i], close_d[i])
        PPs[i] = PP
        S1s[i] = S1
        R1s[i] = R1
    
    # Calculate 1d ADX (14-period)
    high_d_series = pd.Series(high_d)
    low_d_series = pd.Series(low_d)
    close_d_series = pd.Series(close_d)
    
    # True Range
    tr1 = high_d_series - low_d_series
    tr2 = abs(high_d_series - close_d_series.shift(1))
    tr3 = abs(low_d_series - close_d_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_d_series.diff()
    down_move = -low_d_series.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr_d)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr_d)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_d = dx.rolling(window=14, min_periods=14).mean()
    
    # Align Camarilla pivots and ADX to 4h timeframe
    PP_aligned = align_htf_to_ltf(prices, df_daily, PPs)
    S1_aligned = align_htf_to_ltf(prices, df_daily, S1s)
    R1_aligned = align_htf_to_ltf(prices, df_daily, R1s)
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx_d.values)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for 20-period volume average
        # Skip if data not ready
        if (np.isnan(PP_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
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
            # Long: Price breaks above R1 with strong ADX and volume
            if (close[i] > R1_aligned[i] and 
                adx_aligned[i] > 25 and  # Strong trend
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with strong ADX and volume
            elif (close[i] < S1_aligned[i] and 
                  adx_aligned[i] > 25 and  # Strong trend
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below R1 OR ADX weakens
                if close[i] < R1_aligned[i] or adx_aligned[i] < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above S1 OR ADX weakens
                if close[i] > S1_aligned[i] or adx_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1S1_Breakout_1dADX_Volume"
timeframe = "4h"
leverage = 1.0
#%%