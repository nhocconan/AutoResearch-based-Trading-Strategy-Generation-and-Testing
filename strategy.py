#!/usr/bin/env python3
# 6h_1d_elder_ray_volume_trend_v1
# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with volume confirmation and 1-day trend filter.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long: Bull Power > 0 AND increasing AND volume > 1.3x 20-period average volume AND price > 1-day EMA50.
# Short: Bear Power > 0 AND increasing AND volume > 1.3x 20-period average volume AND price < 1-day EMA50.
# Exit: Opposite Elder Ray power > 0 with volume confirmation.
# Designed to capture institutional buying/selling pressure with trend alignment for both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elder_ray_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray calculation
    ema13 = np.full(n, np.nan)
    if n >= 13:
        ema13[12] = np.mean(close[:13])
        for i in range(13, n):
            ema13[i] = close[i] * (2/14) + ema13[i-1] * (12/14)
    
    # Elder Ray components
    bull_power = np.full(n, np.nan)
    bear_power = np.full(n, np.nan)
    for i in range(13, n):
        bull_power[i] = high[i] - ema13[i]
        bear_power[i] = ema13[i] - low[i]
    
    # 20-period average volume
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d_50[i] = close_1d[i] * (2/51) + ema_1d_50[i-1] * (49/51)
    
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        bp = bull_power[i]
        br = bear_power[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_1d = ema_1d_50_aligned[i]
        
        if np.isnan(bp) or np.isnan(br) or np.isnan(avg_vol) or np.isnan(ema_1d):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        vol_surge = vol > 1.3 * avg_vol
        
        # Rising Bull Power (current > previous)
        bull_rising = i > 13 and bp > bull_power[i-1]
        # Rising Bear Power (current > previous)
        bear_rising = i > 13 and br > bear_power[i-1]
        
        if position == 1:  # Long position
            if bear_power[i] > 0 and bear_rising and vol_surge:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if bull_power[i] > 0 and bull_rising and vol_surge:
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = -0.25
        else:  # Flat
            if bull_power[i] > 0 and bull_rising and vol_surge and price > ema_1d:
                position = 1
                signals[i] = 0.25
            elif bear_power[i] > 0 and bear_rising and vol_surge and price < ema_1d:
                position = -1
                signals[i] = -0.25
    
    return signals