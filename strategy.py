#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout with 1-week EMA50 trend filter and volume spike (>2.0x 20-bar MA). Uses weekly HTF trend to capture major market direction (bull/bear regimes) while using Camarilla levels for precise structure-based entries on 4h. Volume confirmation reduces false breakouts. Designed for 15-35 trades/year (60-140 total over 4 years) to minimize fee drag. Weekly trend filter avoids counter-trend trades in both bull and bear markets by only allowing longs in weekly uptrend and shorts in weekly downtrend.
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
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Previous day's OHLC for Camarilla levels (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3, R2, R1, PP, S1, S2, S3
    # R4 = Close + ((High-Low) * 1.1/2)
    # R3 = Close + ((High-Low) * 1.1/4)
    # R2 = Close + ((High-Low) * 1.1/6)
    # R1 = Close + ((High-Low) * 1.1/12)
    # PP = (High + Low + Close) / 3
    # S1 = Close - ((High-Low) * 1.1/12)
    # S2 = Close - ((High-Low) * 1.1/6)
    # S3 = Close - ((High-Low) * 1.1/4)
    # S4 = Close - ((High-Low) * 1.1/2)
    
    rng = high_1d - low_1d
    camarilla_r3 = close_1d + (rng * 1.1 / 4)
    camarilla_s3 = close_1d - (rng * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (they change only at 1d boundaries)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average (tighter than 1.5x)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (20 for vol, 50 for ema, 1 for camarilla)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            bars_since_entry += 1 if position != 0 else 0
            continue
        
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 1w trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1w = close_val > ema_50_val
        bearish_1w = close_val < ema_50_val
        
        # Entry conditions: breakout of Camarilla R3/S3 in trend direction with volume spike
        long_entry = (close_val > camarilla_r3_val) and bullish_1w and vol_spike
        short_entry = (close_val < camarilla_s3_val) and bearish_1w and vol_spike
        
        # Exit conditions: opposite Camarilla level touch (S3 for long, R3 for short)
        exit_long = close_val < camarilla_s3_val
        exit_short = close_val > camarilla_r3_val
        
        # Minimum holding period: 4 bars (to avoid whipsaw)
        min_hold = 4
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -base_size
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = base_size
                bars_since_entry += 1
        elif position == -1:
            # Short - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -base_size
                bars_since_entry += 1
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0