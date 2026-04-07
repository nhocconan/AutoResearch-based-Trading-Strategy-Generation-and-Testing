#!/usr/bin/env python3
"""
4h_camarilla_pivot_1d_trend_volume_v2
Hypothesis: Camarilla pivot levels from 1d act as key intraday support/resistance. 
Price touching S3/R3 with volume confirmation and daily trend alignment provides high-probability reversal entries.
In ranging markets, fade extreme levels; in trending markets, wait for pullbacks to S1/R1.
Targets 20-40 trades/year by requiring confluence of Camarilla touch, volume, and trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Using typical Camarilla formulas based on previous day's range
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: 
    # S3 = C - (H-L)*1.1/2
    # S2 = C - (H-L)*1.1/4
    # S1 = C - (H-L)*1.1/6
    # R1 = C + (H-L)*1.1/12
    # R2 = C + (H-L)*1.1/6
    # R3 = C + (H-L)*1.1/4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    camarilla_s2 = prev_close - (prev_high - prev_low) * 1.1 / 4
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 6
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_r2 = prev_close + (prev_high - prev_low) * 1.1 / 6
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    
    # Align to 4h (these levels are valid for the entire day)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    
    # 20-period EMA for trend filter on 1d
    ema20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False).mean().values
    ema20_4h = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # 20-period SMA for volume average on 4h
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(ema20_4h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S1 (take profit) or breaks below S3 (stop) or trend turns down
            if close[i] <= camarilla_s1_aligned[i] or close[i] < camarilla_s3_aligned[i] or close[i] < ema20_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches R1 (take profit) or breaks above R3 (stop) or trend turns up
            if close[i] >= camarilla_r1_aligned[i] or close[i] > camarilla_r3_aligned[i] or close[i] > ema20_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price touches S3 with volume and uptrend (bounce from strong support)
            if (abs(close[i] - camarilla_s3_aligned[i]) < 0.001 * camarilla_s3_aligned[i] and  # within 0.1% of S3
                vol_confirm and 
                close[i] > ema20_4h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price touches R3 with volume and downtrend (rejection from strong resistance)
            elif (abs(close[i] - camarilla_r3_aligned[i]) < 0.001 * camarilla_r3_aligned[i] and  # within 0.1% of R3
                  vol_confirm and 
                  close[i] < ema20_4h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals