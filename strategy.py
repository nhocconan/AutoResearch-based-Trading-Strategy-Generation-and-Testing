#!/usr/bin/env python3
# 4h_1d_camarilla_pivot_volume_v1
# Strategy: 4h Camarilla pivot reversal with volume confirmation and 1d trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (L3/L4 for longs, H3/H4 for shorts) act as support/resistance.
# In range-bound markets, price reverses at these levels with volume confirmation.
# Uses 1d EMA50 for trend filter to avoid counter-trend trades.
# Designed for low frequency (~20-40/year) to minimize fee drift.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_volume_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily Camarilla pivot levels from previous day
    # Using previous day's high, low, close to avoid look-ahead
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels
    R4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    R3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    R2 = prev_close + ((prev_high - prev_low) * 1.1 / 6)
    R1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    S1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    S2 = prev_close - ((prev_high - prev_low) * 1.1 / 6)
    S3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    S4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align to 4h timeframe (these levels are valid until next day's calculation)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price relative to 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: Camarilla reversal + volume + trend alignment
        # Long when price touches/surpasses S3/S4 in uptrend with volume
        if (close[i] <= S3_aligned[i] and uptrend and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short when price touches/surpasses R3/R4 in downtrend with volume
        elif (close[i] >= R3_aligned[i] and downtrend and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: trend reversal
        elif position == 1 and not uptrend:
            position = 0
            signals[i] = 0.0
        elif position == -1 and not downtrend:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals