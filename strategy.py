#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_volume_v1
# Strategy: 1d Camarilla pivot levels (S3/S4 and R3/R4) with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels act as strong support/resistance. Long near S3/S4 with weekly uptrend,
# short near R3/R4 with weekly downtrend. Volume confirmation filters false breaks. Designed for low trade frequency (~10-25/year) to minimize fee drift.
# Works in bull markets via buying dips at support and bear markets via selling rallies at resistance.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily high, low, close for Camarilla calculation
    high_1d = df_1w['high'].values
    low_1d = df_1w['low'].values
    close_1d = df_1w['close'].values
    
    # Calculate Camarilla levels for previous weekly bar
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.0 * (high - low)
    # S3 = close - 1.0 * (high - low)
    # S4 = close - 1.5 * (high - low)
    hl_range = high_1d - low_1d
    r4 = close_1d + 1.5 * hl_range
    r3 = close_1d + 1.0 * hl_range
    s3 = close_1d - 1.0 * hl_range
    s4 = close_1d - 1.5 * hl_range
    
    # Align Camarilla levels to daily timeframe (previous weekly bar's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_avg_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or \
           np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend: close > EMA20 = uptrend, close < EMA20 = downtrend
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions with Camarilla levels
        # Long: price crosses above S3/S4 with weekly uptrend and volume confirmation
        if weekly_uptrend and vol_confirm[i] and position != 1:
            if close[i] > s3_aligned[i] or close[i] > s4_aligned[i]:
                position = 1
                signals[i] = 0.25
        # Short: price crosses below R3/R4 with weekly downtrend and volume confirmation
        elif weekly_downtrend and vol_confirm[i] and position != -1:
            if close[i] < r3_aligned[i] or close[i] < r4_aligned[i]:
                position = -1
                signals[i] = -0.25
        # Exit: weekly trend reversal
        elif position == 1 and weekly_downtrend:
            position = 0
            signals[i] = 0.0
        elif position == -1 and weekly_uptrend:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals