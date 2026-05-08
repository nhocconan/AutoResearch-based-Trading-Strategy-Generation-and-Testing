#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 AND 1d EMA34 rising AND volume > 2x 20-period average.
# Short when price breaks below Camarilla S3 AND 1d EMA34 falling AND volume > 2x 20-period average.
# Exit when price crosses back inside the Camarilla H-L range (between H4 and L4).
# This strategy targets high-probability breakouts with strong trend and volume confirmation.
# Camarilla levels provide precise support/resistance. The 1d EMA34 filter ensures alignment with daily trend.
# Volume spike (2x average) confirms institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R3_S3_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = close_prev + 1.1 * (high_prev - low_prev) / 6
    camarilla_s3 = close_prev - 1.1 * (high_prev - low_prev) / 6
    camarilla_h4 = close_prev + 1.1 * (high_prev - low_prev) / 2
    camarilla_l4 = close_prev - 1.1 * (high_prev - low_prev) / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Sufficient warmup for EMA34 and Camarilla
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_rising[i]) or 
            np.isnan(ema34_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3, 1d EMA34 rising, volume filter
            long_cond = (close[i] > camarilla_r3_aligned[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price breaks below Camarilla S3, 1d EMA34 falling, volume filter
            short_cond = (close[i] < camarilla_s3_aligned[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Camarilla L4
            if close[i] < camarilla_l4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Camarilla H4
            if close[i] > camarilla_h4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals