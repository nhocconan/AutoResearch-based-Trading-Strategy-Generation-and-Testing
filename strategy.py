#!/usr/bin/env python3
# 1D_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
# Hypothesis: Daily timeframe strategy using weekly Camarilla R3/S3 breakouts with weekly EMA trend filter and volume spike confirmation.
# Works in bull/bear: Weekly trend filter avoids counter-trend trades, volume confirms breakout strength.
# Weekly timeframe provides stable levels, reducing whipsaw in ranging markets. Targets 15-25 trades/year.

name = "1D_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
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
    
    # Get weekly data for Camarilla calculation and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values for Camarilla calculation
    ph = np.concatenate([[high_1w[0]], high_1w[:-1]])  # previous high
    pl = np.concatenate([[low_1w[0]], low_1w[:-1]])   # previous low
    pc = np.concatenate([[close_1w[0]], close_1w[:-1]]) # previous close
    
    # Calculate Camarilla levels (R3, S3 are the key breakout levels)
    rang = ph - pl
    r3 = pc + 1.1 * rang * 1.1666  # R3 = Close + 1.1 * (High-Low) * 1.1666
    s3 = pc - 1.1 * rang * 1.1666  # S3 = Close - 1.1 * (High-Low) * 1.1666
    
    # Align Camarilla levels to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[0:34])
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = (ema_34_1w[i-1] * 33 + close_1w[i]) / 34
    
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike filter: current volume / 20-period average volume
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
    
    start_idx = max(20, 34)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 AND uptrend (price > EMA34) AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 AND downtrend (price < EMA34) AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 OR trend reversal (price < EMA34)
            if close[i] < s3_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 OR trend reversal (price > EMA34)
            if close[i] > r3_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals