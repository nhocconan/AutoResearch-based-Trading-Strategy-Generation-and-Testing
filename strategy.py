#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Camarilla pivot breakout on 4h with 1d EMA34 trend filter and volume spike confirmation.
# Long when 1d trend up (close > EMA34) and price breaks above R3 with volume > 2x average.
# Short when 1d trend down (close < EMA34) and price breaks below S3 with volume > 2x average.
# Uses Camarilla levels from daily timeframe for institutional-grade support/resistance.
# Volume spike filters out false breakouts. Trend filter reduces whipsaw in ranging markets.
# Target: 20-40 trades/year per symbol with disciplined risk management.

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
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
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
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
    
    # Calculate Camarilla levels from previous day
    # R3 = Close + 1.1*(High - Low)*1.1/2, S3 = Close - 1.1*(High - Low)*1.1/2
    camarilla_r3 = np.full_like(close_1d, np.nan)
    camarilla_s3 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        camarilla_r3[i] = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 2
        camarilla_s3[i] = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 2
    
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need 1d EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close[i] > ema34_1d_aligned[i]
        
        if position == 0:
            # Enter long: 1d trend up + price breaks above R3 + volume spike
            if trend_up and close[i] > camarilla_r3_aligned[i] and volume_ratio[i] > 2.0:
                signals[i] = 0.25
                position = 1
            # Enter short: 1d trend down + price breaks below S3 + volume spike
            elif not trend_up and close[i] < camarilla_s3_aligned[i] and volume_ratio[i] > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: 1d trend turns down or price breaks below S3
            if not trend_up or close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: 1d trend turns up or price breaks above R3
            if trend_up or close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals