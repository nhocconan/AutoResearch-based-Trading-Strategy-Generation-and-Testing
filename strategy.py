#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with 12h trend filter.
# Long when price touches S3 level during bullish 12h trend with rejection candle.
# Short when price touches R3 level during bearish 12h trend with rejection candle.
# Uses close > open for bullish candle confirmation. Target: 50-150 total trades over 4 years.

name = "6h_camarilla_pivot_12h_trend_rev_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    # Previous 12h bar's OHLC for pivot calculation
    ph = df_12h['high'].values
    pl = df_12h['low'].values
    pc = df_12h['close'].values
    po = df_12h['open'].values
    
    # Camarilla levels: R3/S3
    pivot = (ph + pl + pc) / 3
    range_ = ph - pl
    r3 = pivot + (range_ * 1.1 / 2)
    s3 = pivot - (range_ * 1.1 / 2)
    
    # Align to 6h timeframe (shifted by 1 for completed bars only)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    
    # 12h trend filter: bullish/bearish based on close vs open
    trend_bullish = pc > po
    trend_bearish = pc < po
    trend_bullish_aligned = align_htf_to_ltf(prices, df_12h, trend_bullish)
    trend_bearish_aligned = align_htf_to_ltf(prices, df_12h, trend_bearish)
    
    # Rejection candle: close near open (small body) after touching level
    body_size = np.abs(close - open_price)
    candle_range = high - low
    # Avoid division by zero
    candle_range = np.where(candle_range == 0, 1e-10, candle_range)
    body_ratio = body_size / candle_range  # < 0.3 indicates rejection
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have previous bar for confirmation
        # Skip if pivot data not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or stop at opposite level
        if position == 1:  # long position
            # Exit: price touches R3 or trend turns bearish
            if (low[i] <= r3_aligned[i] or 
                trend_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price touches S3 or trend turns bullish
            if (high[i] >= s3_aligned[i] or 
                trend_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for reversal entries at S3/R3 with confirmation
            # Long: price touches S3 during bullish 12h trend with rejection candle
            if (low[i] <= s3_aligned[i] and 
                trend_bullish_aligned[i] and
                body_ratio[i] < 0.3):
                signals[i] = 0.25
                position = 1
            # Short: price touches R3 during bearish 12h trend with rejection candle
            elif (high[i] >= r3_aligned[i] and 
                  trend_bearish_aligned[i] and
                  body_ratio[i] < 0.3):
                signals[i] = -0.25
                position = -1
    
    return signals