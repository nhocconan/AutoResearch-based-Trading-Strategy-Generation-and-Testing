#!/usr/bin/env python3
# 1d_Camarilla_R3_S3_Breakout_WeeklyTrendFilter_Volume
# Hypothesis: Daily chart Camarilla R3/S3 breakouts with weekly trend filter and volume spike.
# Weekly trend filter prevents counter-trend trades, volume spike confirms institutional interest.
# Works in bull/bear: Trend filter avoids losses during reversals, volume confirms breakout strength.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

name = "1d_Camarilla_R3_S3_Breakout_WeeklyTrendFilter_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # Calculate Camarilla levels from previous day (daily timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for Camarilla calculation
    ph = np.concatenate([[high_1d[0]], high_1d[:-1]])  # previous high
    pl = np.concatenate([[low_1d[0]], low_1d[:-1]])   # previous low
    pc = np.concatenate([[close_1d[0]], close_1d[:-1]]) # previous close
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = pc + (ph - pl) * 1.1 / 4
    camarilla_s3 = pc - (ph - pl) * 1.1 / 4
    
    # Align Camarilla levels to daily timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate weekly EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[0:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (ema_20_1w[i-1] * 19 + close_1w[i]) / 20
    
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume spike filter: current volume / 20-day average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure volume MA and weekly EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 AND weekly uptrend (close > weekly EMA20) AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_20_1w_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 AND weekly downtrend (close < weekly EMA20) AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 OR weekly trend reversal (close < weekly EMA20)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 OR weekly trend reversal (close > weekly EMA20)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals