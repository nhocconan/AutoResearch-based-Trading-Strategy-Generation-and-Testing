#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d trend filter and volume confirmation
    # Camarilla R3/S3 levels act as key intraday support/resistance
    # 1d EMA(50) trend filter ensures we trade with higher timeframe momentum
    # Volume > 1.5x 20-period average confirms breakout strength
    # Target: 20-40 trades/year per symbol to avoid fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA(50) for trend
    close_1d_series = pd.Series(close_1d)
    ema_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Get 4h data for Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla pivot levels (R3/S3, R4/S4)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = C + (Range * 1.1/4)
    # R4 = C + (Range * 1.1/2)
    # S3 = C - (Range * 1.1/4)
    # S4 = C - (Range * 1.1/2)
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    r3_4h = close_4h + (range_4h * 1.1 / 4.0)
    r4_4h = close_4h + (range_4h * 1.1 / 2.0)
    s3_4h = close_4h - (range_4h * 1.1 / 4.0)
    s4_4h = close_4h - (range_4h * 1.1 / 2.0)
    
    # Align 4h Camarilla levels to 4h timeframe (no alignment needed - same TF)
    # But we need to shift by 1 bar to avoid look-ahead (use previous bar's levels)
    r3_4h_shifted = np.roll(r3_4h, 1)
    r4_4h_shifted = np.roll(r4_4h, 1)
    s3_4h_shifted = np.roll(s3_4h, 1)
    s4_4h_shifted = np.roll(s4_4h, 1)
    pivot_4h_shifted = np.roll(pivot_4h, 1)
    r3_4h_shifted[0] = np.nan
    r4_4h_shifted[0] = np.nan
    s3_4h_shifted[0] = np.nan
    s4_4h_shifted[0] = np.nan
    pivot_4h_shifted[0] = np.nan
    
    # Volume confirmation: current volume > 1.5x 20-period average
    if n >= 20:
        vol_ma = np.full(n, np.nan)
        for i in range(20, n):
            vol_ma[i] = np.mean(volume[i-20:i])
        vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    else:
        vol_ratio = np.ones(n)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r3_4h_shifted[i]) or np.isnan(r4_4h_shifted[i]) or 
            np.isnan(s3_4h_shifted[i]) or np.isnan(s4_4h_shifted[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1d EMA
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Breakout signals: price breaks R3/S3 with volume expansion AND trend alignment
        breakout_long = (close[i] > r3_4h_shifted[i]) and (vol_ratio[i] > 1.5) and uptrend
        breakout_short = (close[i] < s3_4h_shifted[i]) and (vol_ratio[i] > 1.5) and downtrend
        
        # Exit conditions: return to pivot or opposite extreme
        long_exit = close[i] < pivot_4h_shifted[i]
        short_exit = close[i] > pivot_4h_shifted[i]
        
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and position != -1:
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

name = "4h_1d_camarilla_breakout_trend_v1"
timeframe = "4h"
leverage = 1.0