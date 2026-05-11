#!/usr/bin/env python3
"""
12h_Monthly_Pivot_Breakout_Trend_v1
Hypothesis: Monthly pivot levels act as strong support/resistance on 12h timeframe.
Breakout above R1 with monthly trend up = long; breakdown below S1 with monthly trend down = short.
Uses monthly pivot points (calculated from prior month OHLC) and 12h EMA25 for trend filter.
Volume confirmation reduces false breakouts. Designed for 12-37 trades/year.
Works in bull (breakouts continue) and bear (breakdowns continue) markets.
"""

name = "12h_Monthly_Pivot_Breakout_Trend_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Monthly Data for Pivot Points ===
    df_monthly = get_htf_data(prices, '1M')
    if len(df_monthly) < 2:
        return np.zeros(n)
    
    # Previous month OHLC
    monthly_high = df_monthly['high'].values
    monthly_low = df_monthly['low'].values
    monthly_close = df_monthly['close'].values
    
    # Calculate monthly pivot points: P = (H + L + C)/3
    pivot = (monthly_high + monthly_low + monthly_close) / 3.0
    # Resistance 1: R1 = 2*P - L
    r1 = 2 * pivot - monthly_low
    # Support 1: S1 = 2*P - H
    s1 = 2 * pivot - monthly_high
    
    # Align monthly levels to 12h (wait for month-end close)
    pivot_aligned = align_htf_to_ltf(prices, df_monthly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_monthly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_monthly, s1)
    
    # === 12h Data for Trend and Volume ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA25 for trend filter
    ema25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema25_12h)
    
    # Volume average (20-period) for confirmation
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, 25)  # need enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema25_12h_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above R1 with volume confirmation and uptrend
            if (close[i] > r1_aligned[i] and 
                volume[i] > vol_ma_20_aligned[i] * 1.5 and  # 1.5x average volume
                ema25_12h_aligned[i] < close[i]):  # uptrend: price above EMA
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S1 with volume confirmation and downtrend
            elif (close[i] < s1_aligned[i] and 
                  volume[i] > vol_ma_20_aligned[i] * 1.5 and  # 1.5x average volume
                  ema25_12h_aligned[i] > close[i]):  # downtrend: price below EMA
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: breakdown below pivot or trend reversal
            if close[i] < pivot_aligned[i] or ema25_12h_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: breakout above pivot or trend reversal
            if close[i] > pivot_aligned[i] or ema25_12h_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals