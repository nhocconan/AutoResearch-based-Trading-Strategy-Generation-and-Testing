#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: On 12h timeframe, use weekly trend (via EMA200) as primary filter for breakout direction from weekly pivot levels (R3/S3). Enter long when price breaks above weekly R3 with weekly uptrend and volume spike; short when breaks below weekly S3 with weekly downtrend and volume spike. Exit on opposite breakout or trend reversal. Weekly timeframe reduces noise, volume confirms breakout strength, trend filter avoids counter-trend trades. Designed for low trade frequency (<40/year) to minimize fee drag while capturing major moves in both bull and bear markets.

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # Get weekly data for pivot calculation, trend filter, and volume
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Previous week's values for Camarilla calculation
    ph = np.concatenate([[high_1w[0]], high_1w[:-1]])  # previous high
    pl = np.concatenate([[low_1w[0]], low_1w[:-1]])   # previous low
    pc = np.concatenate([[close_1w[0]], close_1w[:-1]]) # previous close
    
    # Calculate Camarilla levels (R3, S3 are the outer breakout levels)
    rang = ph - pl
    r3 = pc + 1.1 * rang * 1.1666  # R3 = Close + 1.1 * (High-Low) * 1.1666
    s3 = pc - 1.1 * rang * 1.1666  # S3 = Close - 1.1 * (High-Low) * 1.1666
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate weekly EMA200 for trend filter
    ema_200_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 200:
        ema_200_1w[199] = np.mean(close_1w[0:200])
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = (ema_200_1w[i-1] * 199 + close_1w[i]) / 200
    
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume spike filter: current weekly volume / 20-period average weekly volume
    vol_ma_1w = np.full_like(volume_1w, np.nan)
    if len(volume_1w) >= 20:
        vol_ma_1w[19] = np.mean(volume_1w[0:20])
        for i in range(20, len(volume_1w)):
            vol_ma_1w[i] = (vol_ma_1w[i-1] * 19 + volume_1w[i]) / 20
    
    volume_ratio_1w = np.full_like(volume_1w, np.nan)
    valid = (~np.isnan(vol_ma_1w)) & (vol_ma_1w != 0)
    volume_ratio_1w[valid] = volume_1w[valid] / vol_ma_1w[valid]
    
    # Align volume ratio to 12h timeframe
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1w, volume_ratio_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(volume_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 AND weekly uptrend (close > EMA200) AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_200_1w_aligned[i] and 
                volume_ratio_aligned[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 AND weekly downtrend (close < EMA200) AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_200_1w_aligned[i] and 
                  volume_ratio_aligned[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 OR trend reversal (close < EMA200)
            if close[i] < s3_aligned[i] or close[i] < ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 OR trend reversal (close > EMA200)
            if close[i] > r3_aligned[i] or close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals