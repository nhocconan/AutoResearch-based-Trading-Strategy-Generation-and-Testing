#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Reversal_WeeklyTrend_Filter
Hypothesis: Daily Camarilla R3/S3 reversal with weekly trend filter (price > weekly EMA50 for long bias, < weekly EMA50 for short bias) and volume confirmation (>1.5x 20-day mean volume). Uses HTF 1w for trend alignment to avoid counter-trend trades in strong weekly trends. Reversal logic captures mean reversion at extreme intraday levels while respecting the weekly trend. Designed for 10-20 trades/year per symbol, effective in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
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
    
    # Get weekly data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on weekly for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align EMA50 to daily timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from previous daily bar (HLC of prior bar)
    camarilla_r3 = close + 1.1 * (high - low)  # R3 = C + 1.1*(H-L)
    camarilla_s3 = close - 1.1 * (high - low)  # S3 = C - 1.1*(H-L)
    
    # Shift to use previous bar's levels (avoid look-ahead)
    camarilla_r3_prev = np.roll(camarilla_r3, 1)
    camarilla_s3_prev = np.roll(camarilla_s3, 1)
    camarilla_r3_prev[0] = np.nan
    camarilla_s3_prev[0] = np.nan
    
    # Volume confirmation: current volume > 1.5x 20-day mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA and volume mean
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(camarilla_r3_prev[i]) or 
            np.isnan(camarilla_s3_prev[i]) or
            np.isnan(vol_mean_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price reverses down from Camarilla R3 in uptrend (price < weekly EMA50 for long bias?)
            # Wait - correction: In uptrend (price > weekly EMA50), we look for reversals DOWN from R3
            # Actually, for mean reversion in uptrend: wait for pullback to S3/S4 levels
            # Simpler: Long when price touches S3 and weekly trend is up
            # Short when price touches R3 and weekly trend is down
            long_signal = (close[i] <= camarilla_s3_prev[i]) and (close[i] > ema_50_aligned[i]) and vol_confirm[i]
            short_signal = (close[i] >= camarilla_r3_prev[i]) and (close[i] < ema_50_aligned[i]) and vol_confirm[i]
            
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
            # Exit when price reverses back to mean (touch S3 again or trend change)
            exit_signal = (close[i] >= camarilla_s3_prev[i]) or (close[i] < ema_50_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price reverses back to mean (touch R3 again or trend change)
            exit_signal = (close[i] <= camarilla_r3_prev[i]) or (close[i] > ema_50_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_Pivot_Reversal_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0