#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R3/S3 breakout on 4h with 1d EMA34 trend filter and volume spike.
# Long when 1d trend up and price breaks above R3 with volume > 2x average.
# Short when 1d trend down and price breaks below S3 with volume > 2x average.
# Uses daily trend to filter whipsaw, volume for confirmation, Camarilla for structure.
# Target: 25-50 trades/year per symbol with disciplined risk management.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema34_1d[i] = (close_1d[i] * 2 + ema34_1d[i-1] * 32) / 34
    
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate previous day's Camarilla levels
    camarilla_r3 = np.full_like(close, np.nan)
    camarilla_s3 = np.full_like(close, np.nan)
    
    # Use previous day's OHLC for today's Camarilla levels
    for i in range(1, len(df_1d)):
        if i >= len(prices):
            break
        # Previous day's OHLC
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        # Camarilla calculations
        range_ = prev_high - prev_low
        camarilla_r3_val = prev_close + range_ * 1.1 / 4
        camarilla_s3_val = prev_close - range_ * 1.1 / 4
        
        # Apply to all 4h bars within this day
        day_start_idx = i * 6  # 6 four-hour bars per day
        day_end_idx = min((i + 1) * 6, n)
        camarilla_r3[day_start_idx:day_end_idx] = camarilla_r3_val
        camarilla_s3[day_start_idx:day_end_idx] = camarilla_s3_val
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 6)  # Need 1d EMA and at least one Camarilla value
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close[i] > ema34_1d_aligned[i]
        
        if position == 0:
            # Enter long: 1d trend up + price breaks above R3 + volume confirmation
            if trend_up and close[i] > camarilla_r3[i] and volume_ratio[i] > 2.0:
                signals[i] = 0.25
                position = 1
            # Enter short: 1d trend down + price breaks below S3 + volume confirmation
            elif not trend_up and close[i] < camarilla_s3[i] and volume_ratio[i] > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: 1d trend turns down or price breaks below S3
            if not trend_up or close[i] < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: 1d trend turns up or price breaks above R3
            if trend_up or close[i] > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals