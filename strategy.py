#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Breakout with Weekly Trend Filter and Volume Spike
# Camarilla pivot levels (R3, S3) represent stronger support/resistance than R1/S1
# Breakouts from these levels with volume confirmation capture stronger momentum moves
# Weekly EMA50 trend filter ensures we only trade in direction of higher timeframe trend
# Works in both bull and bear markets by aligning with weekly trend while capturing 4h breakouts
# Target: 25-35 trades/year (100-140 total over 4 years)

name = "4h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for weekly calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Load daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Camarilla pivot levels (R3, S3) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r3 = close_1d + (1.1 * (high_1d - low_1d) / 4.0)
    s3 = close_1d - (1.1 * (high_1d - low_1d) / 4.0)
    
    # Align daily Camarilla levels to 4h timeframe (completed 1d bar only)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 2.0x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30, 20)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Determine trend: price above weekly EMA50 = uptrend, below = downtrend
        uptrend = curr_close > curr_ema_50_1w
        downtrend = curr_close < curr_ema_50_1w
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and in trend direction
            if curr_volume_confirm:
                # Bullish breakout: price breaks above R3 with volume in uptrend
                if uptrend and curr_close > curr_r3:
                    signals[i] = 0.30
                    position = 1
                # Bearish breakout: price breaks below S3 with volume in downtrend
                elif downtrend and curr_close < curr_s3:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to pivot level OR weekly trend turns down
            if curr_close <= pivot_aligned[i] or curr_close < curr_ema_50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to pivot level OR weekly trend turns up
            if curr_close >= pivot_aligned[i] or curr_close > curr_ema_50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals