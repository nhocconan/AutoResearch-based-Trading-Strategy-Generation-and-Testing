#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_VolumeSpike_Breakout
Hypothesis: Camarilla R3/S3 levels act as strong support/resistance on 6h timeframe.
Breakouts with volume confirmation (volume > 1.5x 20-period MA) provide high-probability entries.
Only trade in direction of weekly EMA20 trend filter to avoid counter-trend whipsaws.
Designed for low trade frequency (~10-20/year) with discrete sizing (0.25) to minimize fee drag.
Works in bull/bear markets via trend alignment - avoids choppy, counter-trend breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla uses previous day's OHLC to calculate today's levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_open = df_1d['open'].shift(1).values
    
    # True range for Camarilla calculation
    tr = np.maximum(prev_high - prev_low, 
                    np.maximum(np.abs(prev_high - prev_close),
                               np.abs(prev_low - prev_close)))
    
    # Camarilla levels (R3, S3, R4, S4)
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    # R4 = close + (high - low) * 1.1/2
    # S4 = close - (high - low) * 1.1/2
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (1-bar delay for HTF close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Weekly EMA20 for trend filter
    weekly_close = df_1w['close'].values
    ema20_1w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for volume MA (20) and weekly EMA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema20_1w_aligned[i]) or
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals with volume confirmation and trend filter
            # Long: price breaks above R3 with volume spike in weekly uptrend (close > weekly EMA20)
            # Short: price breaks below S3 with volume spike in weekly downtrend (close < weekly EMA20)
            long_signal = (close[i] > camarilla_r3_aligned[i]) and volume_spike[i] and (close[i] > ema20_1w_aligned[i])
            short_signal = (close[i] < camarilla_s3_aligned[i]) and volume_spike[i] and (close[i] < ema20_1w_aligned[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below R3 (failed breakout) or weekly trend reverses
            exit_signal = (close[i] < camarilla_r3_aligned[i]) or (close[i] < ema20_1w_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above S3 (failed breakout) or weekly trend reverses
            exit_signal = (close[i] > camarilla_s3_aligned[i]) or (close[i] > ema20_1w_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_VolumeSpike_Breakout"
timeframe = "6h"
leverage = 1.0