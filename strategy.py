#!/usr/bin/env python3
# 12h_Adaptive_Camarilla_Breakout_1wTrend
# Hypothesis: Combine Camarilla pivot levels (R3/S3) from daily timeframe with weekly trend filter (price > 1w EMA50 for long, < for short) and volume confirmation (>1.5x 20-period average). Enter on breakout of R3 (long) or S3 (short) with volume and trend alignment. Exit on opposite crossover or volume drop. Designed for 15-25 trades/year with strict entry conditions to avoid overtrading and capture meaningful moves in both bull and bear markets via trend-following breakouts.

name = "12h_Adaptive_Camarilla_Breakout_1wTrend"
timeframe = "12h"
leverage = 1.0

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
    
    # Load daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for previous day (using prior day's OHLC to avoid look-ahead)
    # We'll use the previous completed day's data for today's levels
    # Since we're on 12h chart, we need to shift the daily data by 1 day for look-ahead safety
    # But align_htf_to_ltf will handle the alignment, so we use the raw daily data and let alignment handle timing
    
    # Calculate typical Camarilla levels based on previous day's range
    # For each day, we calculate R3, S3 based on (H-L) of that day
    # Then we shift these levels forward by 1 day so today's levels are based on yesterday's action
    # This ensures no look-ahead: today's trading levels are based on yesterday's completed range
    
    # First calculate raw Camarilla levels for each day
    ranges = high_1d - low_1d
    close_prev = np.roll(close_1d, 1)  # yesterday's close
    close_prev[0] = close_1d[0]  # first day uses its own close
    
    # Camarilla R3 and S3 formulas
    r3 = close_prev + 1.1 * (high_1d - low_1d) / 2  # Actually: Close + 1.1*(H-L)/2
    s3 = close_prev - 1.1 * (high_1d - low_1d) / 2  # Close - 1.1*(H-L)/2
    
    # Alternative common formula: R3 = Close + 1.1*(H-L), S3 = Close - 1.1*(H-L)
    # Let's use the more common version that matches literature
    r3 = close_prev + 1.1 * (high_1d - low_1d)
    s3 = close_prev - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema50_val = ema50_1w_aligned[i]
        vol_confirm = volume_confirm[i]
        current_close = close[i]
        
        if position == 0:
            # LONG: Price breaks above R3 with volume confirmation and weekly uptrend
            if current_close > r3_val and vol_confirm and current_close > ema50_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume confirmation and weekly downtrend
            elif current_close < s3_val and vol_confirm and current_close < ema50_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 (reversal signal) or volume dries up
            if current_close < s3_val or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 (reversal signal) or volume dries up
            if current_close > r3_val or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals