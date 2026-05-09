# 6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v2
# Hypothesis: Breakout at stronger Camarilla R3/S3 levels with 12h trend filter and volume spike.
# R3/S3 represent stronger support/resistance than R1/S1, reducing false breakouts.
# 12h trend filter provides intermediate-term direction, better for 6h timeframe.
# Volume spike (>2x 20-period average) confirms breakout strength.
# Designed for low trade frequency (12-37/year) to minimize fee drag.
# Works in both bull and bear markets by following the 12h trend direction.

name = "6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v2"
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
    
    # Get 12h data for Camarilla calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous day's values for Camarilla calculation (using 12h bars)
    ph = np.concatenate([[high_12h[0]], high_12h[:-1]])  # previous high
    pl = np.concatenate([[low_12h[0]], low_12h[:-1]])   # previous low
    pc = np.concatenate([[close_12h[0]], close_12h[:-1]]) # previous close
    
    # Calculate Camarilla levels (R3, S3 are stronger breakout levels)
    rang = ph - pl
    r3 = pc + 1.1 * rang * 1.2500  # R3 = Close + 1.1 * (High-Low) * 1.2500
    s3 = pc - 1.1 * rang * 1.2500  # S3 = Close - 1.1 * (High-Low) * 1.2500
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[0:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (ema_50_12h[i-1] * 49 + close_12h[i]) / 50
    
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike filter: current volume / 20-period average volume
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
    bars_since_entry = 0
    
    start_idx = max(20, 50)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price breaks above R3 AND uptrend (price > EMA50) AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price breaks below S3 AND downtrend (price < EMA50) AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 4 bars
            if bars_since_entry < 4:
                signals[i] = 0.25
            else:
                # Exit long: price breaks below S3 OR trend reversal (price < EMA50)
                if close[i] < s3_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 4 bars
            if bars_since_entry < 4:
                signals[i] = -0.25
            else:
                # Exit short: price breaks above R3 OR trend reversal (price > EMA50)
                if close[i] > r3_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals