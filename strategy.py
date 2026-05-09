# 6H_Camarilla_R3_S3_Breakout_1DTrend_VolumeSpike
# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter and volume spike
# Works in bull/bear by requiring 1d trend alignment and volume confirmation to avoid false breakouts
# Target: 50-150 total trades over 4 years (12-37/year)
# Uses 1d trend (EMA34) and volume spike (1.5x 20-period average) as filters
# Entry only when price breaks R3/S3 with trend and volume confirmation
# Exit when price reverses to opposite Camarilla level or trend changes

#!/usr/bin/env python3
name = "6H_Camarilla_R3_S3_Breakout_1DTrend_VolumeSpike"
timeframe = "6h"
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
    
    # Get daily data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1-day EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema34_1d[i] = (close_1d[i] * 2 + ema34_1d[i-1] * 32) / 34
    
    # Align daily EMA34 to 6h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate daily Camarilla levels (based on previous day's range)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    camarilla_r3 = np.full_like(close_1d_prev, np.nan)
    camarilla_s3 = np.full_like(close_1d_prev, np.nan)
    
    for i in range(len(df_1d)):
        if i > 0:  # Need previous day's data
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d_prev[i-1]
            range_hl = prev_high - prev_low
            camarilla_r3[i] = prev_close + 1.1 * range_hl
            camarilla_s3[i] = prev_close - 1.1 * range_hl
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get daily volume for volume spike filter
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        vol_ma20_1d[19] = np.mean(volume_1d[0:20])
        for i in range(20, len(volume_1d)):
            vol_ma20_1d[i] = (volume_1d[i] + vol_ma20_1d[i-1] * 19) / 20
    
    # Align daily volume MA20 to 6h timeframe
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(33, 19)  # Need EMA34 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        daily_trend_up = close[i] > ema34_1d_aligned[i]
        volume_spike = volume[i] > vol_ma20_1d_aligned[i] * 1.5
        
        if position == 0:
            # Enter long: Uptrend + price breaks above R3 + volume spike
            if daily_trend_up and close[i] > camarilla_r3_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + price breaks below S3 + volume spike
            elif not daily_trend_up and close[i] < camarilla_s3_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price breaks below S3
            if not daily_trend_up or close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR price breaks above R3
            if daily_trend_up or close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals