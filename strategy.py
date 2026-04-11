#!/usr/bin/env python3
# 6h_1w_camarilla_breakout_volume_v1
# Strategy: 6h Camarilla pivot breakout from daily levels with weekly trend filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Camarilla levels provide precise support/resistance from prior daily action.
# Breaks above R3 or below S3 with volume confirmation indicate institutional interest.
# Weekly trend filter ensures we only trade in direction of higher timeframe momentum.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue).
# Target: 15-30 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Pre-calculate daily Camarilla levels for each day
    # Arrays to store daily Camarilla levels (same length as prices, updated daily)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    # Calculate Camarilla levels for each completed daily bar
    for i in range(len(df_1d)):
        # Get the daily bar's OHLC
        day_high = df_1d['high'].iloc[i]
        day_low = df_1d['low'].iloc[i]
        day_close = df_1d['close'].iloc[i]
        
        # Calculate Camarilla levels
        range_val = day_high - day_low
        camarilla_r3_val = day_close + range_val * 1.1 / 2
        camarilla_s3_val = day_close - range_val * 1.1 / 2
        camarilla_r4_val = day_close + range_val * 1.1
        camarilla_s4_val = day_close - range_val * 1.1
        
        # Find the time range in prices that corresponds to this daily bar
        # Daily bar i corresponds to 6h bars from i*4 to (i+1)*4 - 1
        start_idx = i * 4
        end_idx = min((i + 1) * 4, n)
        
        # Assign Camarilla levels to all 6h bars within this daily period
        camarilla_r3[start_idx:end_idx] = camarilla_r3_val
        camarilla_s3[start_idx:end_idx] = camarilla_s3_val
        camarilla_r4[start_idx:end_idx] = camarilla_r4_val
        camarilla_s4[start_idx:end_idx] = camarilla_s4_val
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup for EMA and Camarilla
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry logic: Camarilla breakout with volume and trend alignment
        # Long: break above R3 with volume in uptrend
        if (close[i] > camarilla_r3[i] and 
            vol_confirm[i] and 
            uptrend and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short: break below S3 with volume in downtrend
        elif (close[i] < camarilla_s3[i] and 
              vol_confirm[i] and 
              downtrend and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price returns to Camarilla center (close of prior day) or trend change
        elif position == 1 and (close[i] < camarilla_s3[i] or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > camarilla_r3[i] or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals