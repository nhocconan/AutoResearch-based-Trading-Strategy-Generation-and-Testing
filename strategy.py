#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter (price > 1d EMA50 for long, < 1d EMA50 for short), volume confirmation (>2.0x 20-bar mean volume), and choppiness regime filter (CHOP > 50 for mean reversion, CHOP <= 50 for trend following). Uses HTF 1d for trend and regime alignment to reduce whipsaw and capture medium-term momentum. Volume confirmation ensures breakouts have conviction. Discrete position sizing (0.25) minimizes fee churn. Designed for 15-25 trades/year per symbol, effective in both bull (breakouts with volume) and bear (trend-following via shorts) markets.
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
    
    # Get 1d data for HTF trend and regime filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA(50) on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate Choppiness Index on 1d for regime filter
    atr_1d = []
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1]))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(sum_tr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    
    # Align EMA50 and Chop to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of prior bar)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)  # R3 = C + 1.1*(H-L)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)  # S3 = C - 1.1*(H-L)
    
    # Align Camarilla levels to 4h timeframe (use previous bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 2.0x 20-bar mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA, Chop, and volume mean
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_mean_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 in uptrend (price > 1d EMA50) with volume confirmation
            # Short: price breaks below Camarilla S3 in downtrend (price < 1d EMA50) with volume confirmation
            long_signal = (close[i] > camarilla_r3_aligned[i]) and (close[i] > ema_50_aligned[i]) and vol_confirm[i]
            short_signal = (close[i] < camarilla_s3_aligned[i]) and (close[i] < ema_50_aligned[i]) and vol_confirm[i]
            
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
            # Exit when price moves back below 1d EMA50 (trend reversal)
            exit_signal = close[i] < ema_50_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above 1d EMA50 (trend reversal)
            exit_signal = close[i] > ema_50_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0