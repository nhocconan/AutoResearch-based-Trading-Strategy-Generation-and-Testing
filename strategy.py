#!/usr/bin/env python3
# 1h_Camarilla_R3_S3_Breakout_4h1dTrend_Volume
# Hypothesis: Camarilla R3/S3 breakout on 1h with 4h/1d EMA34 trend filter and volume confirmation.
# Long when 4h and 1d trend up, price breaks above R3 with volume > 1.5x average.
# Short when 4h and 1d trend down, price breaks below S3 with volume > 1.5x average.
# Uses 4h/1d for signal direction, 1h for entry timing. Target: 15-37 trades/year per symbol.

name = "1h_Camarilla_R3_S3_Breakout_4h1dTrend_Volume"
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
    
    # Get 4h and 1d data for trend filter
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 34 or len(df_1d) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h EMA34 for trend filter
    ema34_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 34:
        ema34_4h[33] = np.mean(close_4h[0:34])
        for i in range(34, len(close_4h)):
            ema34_4h[i] = (close_4h[i] * 2 + ema34_4h[i-1] * 32) / 34
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema34_1d[i] = (close_1d[i] * 2 + ema34_1d[i-1] * 32) / 34
    
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels (based on previous day's range)
    # For 1h timeframe, we use daily high/low/close to calculate Camarilla
    # We need to get daily data aligned to 1h bars
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_R3 = np.full_like(daily_close, np.nan)
    camarilla_S3 = np.full_like(daily_close, np.nan)
    
    for i in range(len(daily_close)):
        if not (np.isnan(daily_high[i]) or np.isnan(daily_low[i]) or np.isnan(daily_close[i])):
            range_val = daily_high[i] - daily_low[i]
            camarilla_R3[i] = daily_close[i] + range_val * 1.1 / 4
            camarilla_S3[i] = daily_close[i] - range_val * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Volume filter: current volume vs 24-period average (1 day of 1h bars)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 24:
        vol_ma[23] = np.mean(volume[0:24])
        for i in range(24, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 23 + volume[i]) / 24
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)  # Need EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check session
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h and 1d trend
        trend_4h_up = close[i] > ema34_4h_aligned[i]
        trend_1d_up = close[i] > ema34_1d_aligned[i]
        
        if position == 0:
            # Enter long: 4h and 1d trend up + price breaks above R3 + volume confirmation
            if trend_4h_up and trend_1d_up and close[i] > camarilla_R3_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.20
                position = 1
            # Enter short: 4h and 1d trend down + price breaks below S3 + volume confirmation
            elif (not trend_4h_up) and (not trend_1d_up) and close[i] < camarilla_S3_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: 4h or 1d trend turns down or price breaks below S3
            if (not trend_4h_up) or (not trend_1d_up) or close[i] < camarilla_S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: 4h or 1d trend turns up or price breaks above R3
            if trend_4h_up or trend_1d_up or close[i] > camarilla_R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals