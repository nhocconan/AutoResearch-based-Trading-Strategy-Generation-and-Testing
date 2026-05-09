#!/usr/bin/env python3
# 1h_Camarilla_R3_S3_Breakout_4hTrend_Volume
# Hypothesis: 1h strategy using 4h trend filter (EMA34) and daily pivot (Camarilla R3/S3) for direction,
# with volume confirmation on breakouts. 1h used only for entry timing to reduce whipsaw.
# Long when 4h trend up and price breaks above R3 with volume > 1.5x average.
# Short when 4h trend down and price breaks below S3 with volume > 1.5x average.
# Uses 1d pivots for structure, 4h for trend, 1h for entry.
# Designed for low trade frequency (target: 60-150/4 years) to avoid fee drag.

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Get 1d data for Camarilla calculation (pivots)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 4h EMA34 for trend filter
    ema34_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 34:
        ema34_4h[33] = np.mean(close_4h[0:34])
        for i in range(34, len(close_4h)):
            ema34_4h[i] = (close_4h[i] * 2 + ema34_4h[i-1] * 32) / 34
    
    # Calculate 1d Camarilla levels (R3, S3) from previous day
    camarilla_r3 = np.full_like(high_1d, np.nan)
    camarilla_s3 = np.full_like(low_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's data
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        camarilla_r3[i] = pc + (ph - pl) * 1.1 / 4
        camarilla_s3[i] = pc - (ph - pl) * 1.1 / 4
    
    # Align 4h trend and 1d pivots to 1h timeframe
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 1, 20)  # Need 4h EMA, 1d Camarilla, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ratio[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend
        trend_up = close[i] > ema34_4h_aligned[i]
        
        if position == 0:
            # Enter long: 4h trend up + price breaks above R3 + volume confirmation
            if trend_up and close[i] > camarilla_r3_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.20
                position = 1
            # Enter short: 4h trend down + price breaks below S3 + volume confirmation
            elif not trend_up and close[i] < camarilla_s3_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: 4h trend turns down or price breaks below S3
            if not trend_up or close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: 4h trend turns up or price breaks above R3
            if trend_up or close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals