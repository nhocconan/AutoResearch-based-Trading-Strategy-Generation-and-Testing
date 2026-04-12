#!/usr/bin/env python3
"""
4h_1d_1w_Camarilla_Breakout_v1
Hypothesis: Combine daily and weekly Camarilla levels to identify strong breakout points in trending markets.
Long when price breaks above weekly R3 with daily confirmation, short when breaks below weekly S3.
Uses volume confirmation and ADX trend filter to avoid false breakouts in ranging markets.
Designed for low trade frequency (target: 20-50 trades/year) with strong trend capture.
Works in bull via buying breakouts, in bear via selling breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Camarilla_Breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for daily Camarilla
    prev_high_1d = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_low_1d = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    prev_close_1d = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    
    # Previous week's OHLC for weekly Camarilla
    prev_high_1w = df_1w['high'].iloc[-2] if len(df_1w) >= 2 else df_1w['high'].iloc[-1]
    prev_low_1w = df_1w['low'].iloc[-2] if len(df_1w) >= 2 else df_1w['low'].iloc[-1]
    prev_close_1w = df_1w['close'].iloc[-2] if len(df_1w) >= 2 else df_1w['close'].iloc[-1]
    
    # Calculate daily Camarilla levels
    range_1d = prev_high_1d - prev_low_1d
    if range_1d <= 0:
        return np.zeros(n)
    
    camarilla_r3_1d = prev_close_1d + range_1d * 1.1 / 2
    camarilla_s3_1d = prev_close_1d - range_1d * 1.1 / 2
    
    # Calculate weekly Camarilla levels
    range_1w = prev_high_1w - prev_low_1w
    if range_1w <= 0:
        return np.zeros(n)
    
    camarilla_r3_1w = prev_close_1w + range_1w * 1.1 / 2
    camarilla_s3_1w = prev_close_1w - range_1w * 1.1 / 2
    
    # Align daily and weekly levels to 4h timeframe
    camarilla_r3_1d_arr = np.full(len(df_1d), camarilla_r3_1d)
    camarilla_s3_1d_arr = np.full(len(df_1d), camarilla_s3_1d)
    camarilla_r3_1w_arr = np.full(len(df_1w), camarilla_r3_1w)
    camarilla_s3_1w_arr = np.full(len(df_1w), camarilla_s3_1w)
    
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d_arr)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d_arr)
    camarilla_r3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w_arr)
    camarilla_s3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w_arr)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values
    
    # ADX trend filter (14-period) - use daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any data invalid
        if (np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i]) or
            np.isnan(camarilla_r3_1w_aligned[i]) or np.isnan(camarilla_s3_1w_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        # Breakout conditions with volume and trend filter
        long_breakout = (high[i] > camarilla_r3_1w_aligned[i] and 
                        close[i] > camarilla_r3_1d_aligned[i] and
                        vol_ratio[i] > 1.5 and
                        adx_aligned[i] > 25)
        
        short_breakout = (low[i] < camarilla_s3_1w_aligned[i] and 
                         close[i] < camarilla_s3_1d_aligned[i] and
                         vol_ratio[i] > 1.5 and
                         adx_aligned[i] > 25)
        
        # Exit conditions: return to opposite Camarilla level or trend weakening
        long_exit = (close[i] < camarilla_s3_1d_aligned[i] or 
                    adx_aligned[i] < 20)
        short_exit = (close[i] > camarilla_r3_1d_aligned[i] or 
                     adx_aligned[i] < 20)
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.30
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals