#!/usr/bin/env python3
# 12h_1w_1d_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: Combines weekly and daily trends with 12-hour Candlestick patterns using Camarilla pivot levels (R3/S3).
# Uses 1-week EMA for long-term trend, 1-day EMA for intermediate trend, and volume spike for confirmation.
# Designed for 12h timeframe to capture multi-day trends while avoiding overtrading.
# Target: 12-37 trades per year per symbol (50-150 total over 4 years) with strong performance in bull and bear regimes.

name = "12h_1w_1d_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Get weekly data for long-term trend and pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for intermediate trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly OHLC for Camarilla pivots
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Camarilla levels from weekly data: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    rng_1w = high_1w - low_1w
    camarilla_r3 = close_1w + rng_1w * 1.1 / 4
    camarilla_s3 = close_1w - rng_1w * 1.1 / 4
    
    # 1-week EMA40 for long-term trend filter
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # Align all indicators to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume filter: current volume > 1.8x average volume (30-period)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 40)  # Ensure we have volume MA and weekly EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema40_1w_aligned[i]) or np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R3 + Uptrend (price > weekly EMA40 AND daily EMA20) + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema40_1w_aligned[i] and
                close[i] > ema20_1d_aligned[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + Downtrend (price < weekly EMA40 AND daily EMA20) + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema40_1w_aligned[i] and
                  close[i] < ema20_1d_aligned[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions:
            # 1. Price returns inside weekly Camarilla range (reversion to mean)
            # 2. Trend reversal: price crosses below/above weekly EMA40
            price_inside = (close[i] < r3_aligned[i] and close[i] > s3_aligned[i])
            trend_reversal = (position == 1 and close[i] < ema40_1w_aligned[i]) or \
                            (position == -1 and close[i] > ema40_1w_aligned[i])
            
            if price_inside or trend_reversal:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals