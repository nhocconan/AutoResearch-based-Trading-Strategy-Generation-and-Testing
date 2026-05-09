#!/usr/bin/env python3
# 1d_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Camarilla pivot levels from daily data (R3/S3) breakout with weekly trend filter and volume confirmation.
# Long when weekly trend up and price breaks above daily R3 with volume > 1.5x average.
# Short when weekly trend down and price breaks below daily S3 with volume > 1.5x average.
# Target: 10-30 trades/year per symbol with disciplined risk management.

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) for each day
    # Using previous day's OHLC
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    # Calculate Camarilla for each day based on previous day's OHLC
    for i in range(len(df_1d)):
        # Get previous day's OHLC (using current day's index for calculation)
        prev_high = df_1d['high'].iloc[i]
        prev_low = df_1d['low'].iloc[i]
        prev_close = df_1d['close'].iloc[i]
        
        # Camarilla formulas
        range_val = prev_high - prev_low
        camarilla_r3_val = prev_close + range_val * 1.1 / 2
        camarilla_s3_val = prev_close - range_val * 1.1 / 2
        
        # Find corresponding indices in the intraday data
        day_start_idx = i * 24  # Assuming 24 hours in a day
        day_end_idx = min((i + 1) * 24, n)
        
        camarilla_r3[day_start_idx:day_end_idx] = camarilla_r3_val
        camarilla_s3[day_start_idx:day_end_idx] = camarilla_s3_val
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema50_1w[i] = (close_1w[i] * 2 + ema50_1w[i-1] * 48) / 50
    
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume vs 24-period average (hourly)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 24:
        vol_ma[23] = np.mean(volume[0:24])
        for i in range(24, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 23 + volume[i]) / 24
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 24)  # Need volume MA and Camarilla levels
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        trend_up = close[i] > ema50_1w_aligned[i]
        
        if position == 0:
            # Enter long: weekly trend up + price breaks above daily R3 + volume confirmation
            if trend_up and close[i] > camarilla_r3[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly trend down + price breaks below daily S3 + volume confirmation
            elif not trend_up and close[i] < camarilla_s3[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend turns down or price breaks below daily S3
            if not trend_up or close[i] < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend turns up or price breaks above daily R3
            if trend_up or close[i] > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals