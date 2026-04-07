#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot Reversal with Weekly Trend Filter
# Hypothesis: Camarilla pivot levels (R3/S3) act as strong reversal zones in ranging markets.
# Weekly trend filter ensures we only take reversals aligned with higher timeframe direction.
# Works in both bull and bear markets by fading extremes in the direction of weekly trend.
# Targets 15-30 trades/year with high-probability mean-reversion entries.

name = "6h_camarilla_pivot_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter: EMA50 on weekly timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We'll use the previous day's close as base for simplicity
    # Actually, Camarilla uses: R3 = Close + (High-Low)*1.1/4, S3 = Close - (High-Low)*1.1/4
    
    # Get previous day's OHLC (already completed daily bar)
    prev_day_close = df_1d['close'].values
    prev_day_high = df_1d['high'].values
    prev_day_low = df_1d['low'].values
    
    # Calculate Camarilla R3 and S3 levels
    rang = prev_day_high - prev_day_low
    r3 = prev_day_close + (rang * 1.1 / 4)
    s3 = prev_day_close - (rang * 1.1 / 4)
    
    # Align to 6h timeframe (these levels are fixed for the entire day)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from 1 to have previous bar data
        # Skip if required data not available
        if (np.isnan(ema50_6h[i]) or 
            np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches S3 level or weekly trend turns bearish
            if close[i] <= s3_6h[i] or close[i] < ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches R3 level or weekly trend turns bullish
            if close[i] >= r3_6h[i] or close[i] > ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry at R3/S3 reversals
            # Long: price touches/bounces off S3 in uptrend
            if (low[i] <= s3_6h[i] and close[i] > s3_6h[i] and 
                close[i] > ema50_6h[i]):  # Weekly uptrend filter
                position = 1
                signals[i] = 0.25
            # Short: price touches/bounces off R3 in downtrend
            elif (high[i] >= r3_6h[i] and close[i] < r3_6h[i] and 
                  close[i] < ema50_6h[i]):  # Weekly downtrend filter
                position = -1
                signals[i] = -0.25
    
    return signals