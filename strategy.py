#!/usr/bin/env python3
# 2025-06-22 | 4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS
# Hypothesis: Camarilla pivot R3/S3 breakout on 4h with 1d EMA trend filter and volume spike.
# Long when price breaks above R3 with volume > 1.5x average and price > 1d EMA34.
# Short when price breaks below S3 with volume > 1.5x average and price < 1d EMA34.
# Uses discrete position sizing (0.25) to limit overtrading and fee drag.
# Designed for 20-50 trades/year on 4h timeframe to avoid fee drag in BTC/ETH.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34)
    ema_34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2 + ema_34_1d[i-1] * 32) / 34
    
    # Align 1d EMA to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Camarilla levels from previous day
    # We need previous day's high, low, close
    # For each 4h bar, we look at the prior day's HLC
    # Since we're on 4h timeframe, we can calculate once per day
    
    # First, get daily OHLC from 1d data
    # But we need to align it to 4h bars - each 4h bar gets the prior day's values
    
    # Calculate Camarilla for each day using 1d data
    camarilla_R3 = np.full_like(close_1d, np.nan)
    camarilla_S3 = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i >= 1:  # Need previous day
            prev_high = df_1d['high'].values[i-1]
            prev_low = df_1d['low'].values[i-1]
            prev_close = df_1d['close'].values[i-1]
            range_val = prev_high - prev_low
            camarilla_R3[i] = prev_close + range_val * 1.1 / 4
            camarilla_S3[i] = prev_close - range_val * 1.1 / 4
        # For first day, leave as NaN
    
    # Align Camarilla levels to 4h timeframe (using prior day's values)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Volume filter: 4h volume / 20-period average volume
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
    
    start_idx = max(20, 1)  # Ensure volume MA and at least 1 day of data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or \
           np.isnan(camarilla_S3_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 AND volume confirmation AND price > 1d EMA
            if close[i] > camarilla_R3_aligned[i] and volume_ratio[i] > 1.5 and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 AND volume confirmation AND price < 1d EMA
            elif close[i] < camarilla_S3_aligned[i] and volume_ratio[i] > 1.5 and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 (reversal) or volume drops
            if close[i] < camarilla_S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 (reversal) or volume drops
            if close[i] > camarilla_R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals