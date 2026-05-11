#!/usr/bin/env python3
"""
12h_1d_RSI_Reversal_With_Trend_Filter
Hypothesis: On 12h timeframe, use RSI(14) extremes for mean reversion entries, filtered by 1d EMA50 trend direction.
- Long when: RSI < 30 and price > 1d EMA50 (oversold in uptrend)
- Short when: RSI > 70 and price < 1d EMA50 (overbought in downtrend)
- Exit when: RSI crosses back to neutral (40 for longs, 60 for shorts)
Uses 1d trend filter to avoid counter-trend trades and RSI extremes for high-probability reversals.
Targets 15-25 trades/year (60-100 over 4 years) to minimize fee drag.
"""

name = "12h_1d_RSI_Reversal_With_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- RSI(14) on 12h ---
    delta = pd.Series(close_12h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and RSI
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close_12h[i] > ema50_1d_aligned[i]
        trend_down = close_12h[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with RSI extremes
            if rsi_values[i] < 30 and trend_up:
                # Oversold in uptrend -> long
                signals[i] = 0.25
                position = 1
            elif rsi_values[i] > 70 and trend_down:
                # Overbought in downtrend -> short
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions based on RSI mean reversion
            if position == 1:
                # Exit long when RSI returns to 40 (recovered from oversold)
                if rsi_values[i] >= 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short when RSI returns to 60 (declined from overbought)
                if rsi_values[i] <= 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals