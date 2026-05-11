# 1d_1w_Camarilla_R3_S3_Breakout_TrendFilter
# Hypothesis: Weekly Camarilla pivot levels (R3, S3) act as strong support/resistance on the daily chart.
# Long when: price breaks above R3 with volume confirmation and weekly trend up (EMA10 > EMA30).
# Short when: price breaks below S3 with volume confirmation and weekly trend down (EMA10 < EMA30).
# Exit when price returns to the weekly pivot (P) or trend reverses.
# Weekly Camarilla levels are derived from prior week's OHLC and represent institutional interest zones.
# Works in bull by buying breakouts above R3 in uptrend; works in bear by selling breakdowns below S3 in downtrend.
# Target: 15-25 trades/year (60-100 total over 4 years) to avoid fee drag.

name = "1d_1w_Camarilla_R3_S3_Breakout_TrendFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for Camarilla levels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly Camarilla levels (based on prior week's OHLC) ---
    # Calculate for each week: using (high, low, close) of that week
    H = df_1w['high'].values
    L = df_1w['low'].values
    C = df_1w['close'].values
    
    # Typical price for pivot calculation
    PP = (H + L + C) / 3  # Pivot point
    RANGE = H - L
    
    # Camarilla levels
    R3 = PP + (RANGE * 1.1 / 2)  # R3 = PP + (Range * 1.1/2)
    S3 = PP - (RANGE * 1.1 / 2)  # S3 = PP - (Range * 1.1/2)
    P = PP                       # Pivot point (for exit)
    
    # --- Weekly EMA10 and EMA30 for trend ---
    close_1w = df_1w['close'].values
    ema10 = np.full(len(close_1w), np.nan)
    ema30 = np.full(len(close_1w), np.nan)
    
    # Calculate EMA10
    for i in range(len(close_1w)):
        if i < 10:
            ema10[i] = np.nan
        elif i == 10:
            ema10[i] = np.mean(close_1w[0:10])
        else:
            ema10[i] = (close_1w[i] * 2 / (10 + 1)) + (ema10[i-1] * (9 / (10 + 1)))
    
    # Calculate EMA30
    for i in range(len(close_1w)):
        if i < 30:
            ema30[i] = np.nan
        elif i == 30:
            ema30[i] = np.mean(close_1w[0:30])
        else:
            ema30[i] = (close_1w[i] * 2 / (30 + 1)) + (ema30[i-1] * (29 / (30 + 1)))
    
    # Trend: 1 if EMA10 > EMA30, -1 if EMA10 < EMA30
    trend = np.zeros(len(close_1w))
    for i in range(len(close_1w)):
        if not np.isnan(ema10[i]) and not np.isnan(ema30[i]):
            if ema10[i] > ema30[i]:
                trend[i] = 1
            elif ema10[i] < ema30[i]:
                trend[i] = -1
    
    # Align weekly indicators to daily
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    P_aligned = align_htf_to_ltf(prices, df_1w, P)
    trend_aligned = align_htf_to_ltf(prices, df_1w, trend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have at least one complete week of data
    start_idx = 0
    for i in range(n):
        if not np.isnan(R3_aligned[i]):
            start_idx = i
            break
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(P_aligned[i]) or
            np.isnan(trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume confirmation and uptrend
            if close[i] > R3_aligned[i] and trend_aligned[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume confirmation and downtrend
            elif close[i] < S3_aligned[i] and trend_aligned[i] < 0:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price returns to pivot (P) OR trend turns down
                if close[i] < P_aligned[i] or trend_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to pivot (P) OR trend turns up
                if close[i] > P_aligned[i] or trend_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals