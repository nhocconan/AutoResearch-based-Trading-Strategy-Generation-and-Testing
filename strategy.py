#!/usr/bin/env python3
# 1h_Camarilla_R3_S3_Breakout_4hTrend_DailyVolumeSpike
# Hypothesis: 1h timeframe strategy using 4h trend (EMA200) and daily volume spike for confirmation.
# In bull markets: Buy breakouts above R3 when 4h trend is up and volume confirms.
# In bear markets: Sell breakouts below S3 when 4h trend is down and volume confirms.
# Uses 1h only for entry timing, 4h for trend direction, daily for volume context.
# Target: 15-37 trades/year (60-150 total over 4 years) with 0.20 position size.

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_DailyVolumeSpike"
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
    
    # Calculate Camarilla levels from previous day (using 1d data)
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
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 4h EMA200 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_200_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 200:
        ema_200_4h[199] = np.mean(close_4h[0:200])
        for i in range(200, len(close_4h)):
            ema_200_4h[i] = (ema_200_4h[i-1] * 199 + close_4h[i]) / 200
    
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Daily volume spike filter: current volume / 20-day average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    # Session filter: 08-20 UTC (avoid low liquidity periods)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_200_4h_aligned[i]) or np.isnan(volume_ratio[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 AND 4h uptrend (price > EMA200) AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_200_4h_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below S3 AND 4h downtrend (price < EMA200) AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_200_4h_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 OR trend reversal (price < EMA200)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above R3 OR trend reversal (price > EMA200)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals