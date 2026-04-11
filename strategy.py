#!/usr/bin/env python3
# 12h_1w_camarilla_pivot_volume_v1
# Strategy: 12h Camarilla pivot levels with weekly trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels provide high-probability reversal zones. 
# In ranging markets, price reverts from S3/R3 levels. In trending markets, 
# breaks of S4/R4 with weekly trend alignment and volume capture strong moves.
# Weekly trend filter avoids counter-trend trades. Target: 15-35 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily data for Camarilla pivot calculation (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas: 
    # R4 = close + ((high - low) * 1.1 / 2)
    # R3 = close + ((high - low) * 1.1 / 4)
    # S3 = close - ((high - low) * 1.1 / 4)
    # S4 = close - ((high - low) * 1.1 / 2)
    rang = high_1d - low_1d
    r4 = close_1d + (rang * 1.1 / 2)
    r3 = close_1d + (rang * 1.1 / 4)
    s3 = close_1d - (rang * 1.1 / 4)
    s4 = close_1d - (rang * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend filter
        uptrend_weekly = close[i] > ema_20_1w_aligned[i]
        downtrend_weekly = close[i] < ema_20_1w_aligned[i]
        
        # Long setup: price at S3/S4 with weekly uptrend and volume
        if (close[i] <= s3_aligned[i] or close[i] <= s4_aligned[i]) and \
           uptrend_weekly and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short setup: price at R3/R4 with weekly downtrend and volume
        elif (close[i] >= r3_aligned[i] or close[i] >= r4_aligned[i]) and \
             downtrend_weekly and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price reaches opposite level or weekly trend changes
        elif position == 1 and (close[i] >= r3_aligned[i] or not uptrend_weekly):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= s3_aligned[i] or not downtrend_weekly):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals