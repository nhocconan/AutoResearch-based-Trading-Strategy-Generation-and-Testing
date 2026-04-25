#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Trade 6h Camarilla R3/S3 breakouts with 1d trend filter (EMA50) and volume spike confirmation.
In bull markets, buy R3 breakouts with volume; in bear markets, sell S3 breakdowns with volume.
The 1d EMA50 filter ensures we only trade with the higher timeframe trend, reducing whipsaws.
Volume spike (volume > 1.5x 20-period average) confirms institutional participation.
Target: 12-37 trades/year per symbol (50-150 total over 4 years) to avoid fee drag on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h Camarilla levels (based on previous day's OHLC)
    # We need to use daily OHLC to calculate Camarilla levels for intraday periods
    # For 6h timeframe, we calculate Camarilla based on previous 1d bar
    # Typical Camarilla formula: 
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    
    # Since we're on 6h timeframe, we need to align the daily Camarilla levels
    # Calculate daily OHLC first
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # Using typical Camarilla multipliers
    camarilla_multiplier = 1.1
    rng = high_1d - low_1d
    
    # Calculate levels
    r4 = close_1d + (rng * camarilla_multiplier / 2)
    r3 = close_1d + (rng * camarilla_multiplier / 4)
    r2 = close_1d + (rng * camarilla_multiplier / 6)
    r1 = close_1d + (rng * camarilla_multiplier / 12)
    pp = (high_1d + low_1d + close_1d) / 3
    s1 = close_1d - (rng * camarilla_multiplier / 12)
    s2 = close_1d - (rng * camarilla_multiplier / 6)
    s3 = close_1d - (rng * camarilla_multiplier / 4)
    s4 = close_1d - (rng * camarilla_multiplier / 2)
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for volume MA (20) and 1d EMA50 (50)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume spike AND 1d trend bullish
            long_setup = (close[i] > r3_aligned[i]) and \
                         volume_spike[i] and \
                         (close_1d_align := align_htf_to_ltf(prices, df_1d, close_1d)[i]) > ema_50_1d_aligned[i]
            # Short: price breaks below S3 with volume spike AND 1d trend bearish
            short_setup = (close[i] < s3_aligned[i]) and \
                          volume_spike[i] and \
                          (close_1d_align := align_htf_to_ltf(prices, df_1d, close_1d)[i]) < ema_50_1d_aligned[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters R3-R4 range OR 1d trend turns bearish
            if (close[i] < r3_aligned[i]) or \
               (align_htf_to_ltf(prices, df_1d, close_1d)[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters S3-S4 range OR 1d trend turns bullish
            if (close[i] > s3_aligned[i]) or \
               (align_htf_to_ltf(prices, df_1d, close_1d)[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0