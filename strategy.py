#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot S1/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above S3 and closes above S3, price > 1d EMA34, volume > 1.5x 20-period average.
# Short when price breaks below S1 and closes below S1, price < 1d EMA34, volume > 1.5x 20-period average.
# Exit when price returns to S1/S3 level or volume filter fails.
# Uses 1d EMA34 for trend filter to avoid counter-trend trades.
# Volume filter ensures participation and avoids low-conviction moves.
# Target: 20-40 trades/year to minimize fee drag.
name = "4h_Camarilla_S1S3_Breakout_1dEMA34_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Previous day's high, low, close for Camarilla calculation
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Camarilla levels: S1 = C - (H-L)*1.12/6, S3 = C - (H-L)*1.12/2
    # R1 = C + (H-L)*1.12/6, R3 = C + (H-L)*1.12/2
    # We focus on S1 and S3 for mean reversion breakouts
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.12 / 6
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.12 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above S3 and closes above S3, uptrend, volume filter
            long_cond = (close[i] > camarilla_s3_aligned[i]) and (close[i-1] <= camarilla_s3_aligned[i-1]) and (close[i] > ema34_1d_aligned[i]) and volume_filter[i]
            # Short conditions: price breaks below S1 and closes below S1, downtrend, volume filter
            short_cond = (close[i] < camarilla_s1_aligned[i]) and (close[i-1] >= camarilla_s1_aligned[i-1]) and (close[i] < ema34_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to S3 level or volume filter fails
            if close[i] <= camarilla_s3_aligned[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to S1 level or volume filter fails
            if close[i] >= camarilla_s1_aligned[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals