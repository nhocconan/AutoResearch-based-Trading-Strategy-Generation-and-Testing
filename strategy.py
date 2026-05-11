#!/usr/bin/env python3
"""
6h_1w_Trix_Zero_Cross_With_RSI
Hypothesis: Uses weekly TRIX (15-period) to capture long-term momentum and zero-line crosses for trend direction.
Enters on 6h when TRIX crosses zero in direction of weekly trend, with RSI(14) > 50 for long or < 50 for short to avoid countertrend whipsaws.
Exits when TRIX crosses back across zero or RSI reverts to neutral zone (40-60).
Designed to work in both bull and bear markets by following higher-timeframe momentum.
Targets low trade frequency (15-30/year) via weekly momentum filter and zero-cross logic.
"""

name = "6h_1w_Trix_Zero_Cross_With_RSI"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_trix(close, period=15):
    """Calculate TRIX: triple EMA then percentage change"""
    # First EMA
    ema1 = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean()
    # Second EMA of first EMA
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    # Third EMA of second EMA
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
    # TRIX = percentage change of third EMA
    trix = ema3.pct_change() * 100
    return trix.values

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly TRIX for Trend Filter ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 45:  # Need enough for TRIX calculation
        return np.zeros(n)
    
    trix_w = calculate_trix(df_1w['close'].values, 15)
    # Weekly trend: 1 if TRIX > 0 (bullish momentum), -1 if TRIX < 0 (bearish momentum)
    trend_w = np.where(trix_w > 0, 1, -1)
    
    # Align weekly TRIX and trend to 6h timeframe
    trix_w_6h = align_htf_to_ltf(prices, df_1w, trix_w)
    trend_w_6h = align_htf_to_ltf(prices, df_1w, trend_w)
    
    # --- 6h RSI for Entry Filter ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = np.nan_to_num(rsi, nan=50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 45
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_w_6h[i]) or np.isnan(trend_w_6h[i]) or 
            np.isnan(rsi[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Weekly TRIX zero-cross detection
        trix_now = trix_w_6h[i]
        trix_prev = trix_w_6h[i-1]
        weekly_trend = trend_w_6h[i]
        rsi_now = rsi[i]
        
        if position == 0:
            # Long: weekly bullish (TRIX > 0) + TRIX crosses above zero + RSI > 50
            if (weekly_trend == 1 and 
                trix_prev <= 0 and trix_now > 0 and  # crossed above zero
                rsi_now > 50):
                signals[i] = 0.25
                position = 1
            # Short: weekly bearish (TRIX < 0) + TRIX crosses below zero + RSI < 50
            elif (weekly_trend == -1 and 
                  trix_prev >= 0 and trix_now < 0 and  # crossed below zero
                  rsi_now < 50):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: TRIX crosses below zero OR RSI drops below 40
                if (trix_prev >= 0 and trix_now < 0) or rsi_now < 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: TRIX crosses above zero OR RSI rises above 60
                if (trix_prev <= 0 and trix_now > 0) or rsi_now > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals