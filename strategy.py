#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm
Hypothesis: Camarilla R3/S3 breakout on 4h with 1d EMA50 trend filter and volume confirmation.
Uses tighter entry (R3/S3 vs R1/S1) to reduce trades from 1700+ to ~100-200/year.
Long when price > R3 and > EMA50 with volume spike. Short when price < S3 and < EMA50 with volume spike.
Exit on trend reversal (close crosses EMA50) or range re-entry (close crosses opposite S1/R1).
Designed for 4h timeframe targeting 100-200 trades over 4 years (25-50/year).
Works in bull/bear markets: EMA50 filter ensures we only trade with the daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla levels from previous 1d bar (completed)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.1 / 12)
    s1 = prev_close - (rng * 1.1 / 12)
    r3 = prev_close + (rng * 1.1 / 4)
    s3 = prev_close - (rng * 1.1 / 4)
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA50 trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume spike: current > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d shift, EMA50, vol avg
    start_idx = max(30, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry: Camarilla R3/S3 breakout with EMA alignment and volume spike
            long_condition = (close_val > r3_val and 
                            close_val > ema_val and 
                            vol_spike)
            short_condition = (close_val < s3_val and 
                             close_val < ema_val and 
                             vol_spike)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend reversal (close < EMA50) OR range re-entry (close < S1)
            if close_val < ema_val or close_val < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend reversal (close > EMA50) OR range re-entry (close > R1)
            if close_val > ema_val or close_val > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0