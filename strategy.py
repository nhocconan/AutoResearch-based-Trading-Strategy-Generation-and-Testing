#!/usr/bin/env python3
# 6h_1d_1w_cci_trend_reversal_v1
# Hypothesis: 6-hour CCI trend reversal with 1-day/1-week trend filter.
# Long when CCI crosses below -100 (oversold) and price above 1-day EMA200 and 1-week EMA50.
# Short when CCI crosses above +100 (overbought) and price below 1-day EMA200 and 1-week EMA50.
# Exit when CCI crosses back above -50 (long) or below +50 (short).
# Uses multi-timeframe alignment to avoid look-ahead and ensure proper trend filtering.
# Designed to generate ~15-30 trades/year to avoid fee decay while capturing mean reversion in trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_cci_trend_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d and 1w data for trend filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 200 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema200_1d[199] = np.mean(close_1d[:200])
        alpha = 2.0 / (200 + 1)
        for i in range(200, len(close_1d)):
            ema200_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema200_1d[i-1]
    
    # Calculate 1-week EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[:50])
        alpha = 2.0 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema50_1w[i-1]
    
    # Calculate CCI (20-period)
    cci = np.full(n, np.nan)
    if n >= 20:
        tp = (high + low + close) / 3.0  # Typical price
        sma_tp = np.full(n, np.nan)
        mad = np.full(n, np.nan)
        
        # Calculate SMA of typical price
        for i in range(19, n):
            sma_tp[i] = np.mean(tp[i-19:i+1])
        
        # Calculate Mean Absolute Deviation
        for i in range(19, n):
            deviation = np.abs(tp[i-19:i+1] - sma_tp[i])
            mad[i] = np.mean(deviation)
        
        # Calculate CCI
        for i in range(19, n):
            if mad[i] != 0:
                cci[i] = (tp[i] - sma_tp[i]) / (0.015 * mad[i])
    
    # Align 1d EMA200 and 1w EMA50 to 6h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(cci[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        cci_val = cci[i]
        ema200_val = ema200_1d_aligned[i]
        ema50_w_val = ema50_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: CCI crosses back above -50 (exiting oversold)
            if cci_val > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: CCI crosses back below +50 (exiting overbought)
            if cci_val < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: CCI extreme with trend filter
            # Enter long: CCI crosses below -100 (oversold) and price above both EMAs
            if cci_val < -100 and price > ema200_val and price > ema50_w_val:
                position = 1
                signals[i] = 0.25
            # Enter short: CCI crosses above +100 (overbought) and price below both EMAs
            elif cci_val > 100 and price < ema200_val and price < ema50_w_val:
                position = -1
                signals[i] = -0.25
    
    return signals