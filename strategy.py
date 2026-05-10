#!/usr/bin/env python3
# 6H_1W1D_Camarilla_Reversal_Trend
# Hypothesis: Use weekly trend direction for bias, daily Camarilla R3/S3 for fade entries in mean-reversion zones, 
# and 6H close breaks above/below R4/S4 for trend continuation. Combines mean-reversion and trend-following 
# with clear risk control. Works in bull/bear by following weekly trend while using daily structure for entries.

name = "6H_1W1D_Camarilla_Reversal_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Weekly trend: bullish if close > EMA34, bearish if close < EMA34
    bullish_trend_1w = close_1w > ema34_1w
    bearish_trend_1w = close_1w < ema34_1w
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # S4 = C - ((H-L) * 1.1/2), S3 = C - ((H-L) * 1.1/4)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day has no previous
    prev_high[0] = prev_high[1] if len(prev_high) > 1 else high_1d[0]
    prev_low[0] = prev_low[1] if len(prev_low) > 1 else low_1d[0]
    prev_close[0] = prev_close[1] if len(prev_close) > 1 else close_1d[0]
    
    rang = prev_high - prev_low
    r4 = prev_close + rang * 1.1 / 2
    r3 = prev_close + rang * 1.1 / 4
    s3 = prev_close - rang * 1.1 / 4
    s4 = prev_close - rang * 1.1 / 2
    
    # Align weekly trend to 6h
    bullish_1w_aligned = align_htf_to_ltf(prices, df_1w, bullish_trend_1w.astype(float))
    bearish_1w_aligned = align_htf_to_ltf(prices, df_1w, bearish_trend_1w.astype(float))
    
    # Align daily Camarilla levels to 6h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bullish_1w_aligned[i]) or np.isnan(bearish_1w_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish_weekly = bullish_1w_aligned[i] > 0.5
        bearish_weekly = bearish_1w_aligned[i] > 0.5
        
        if position == 0:
            # Mean-reversion fade at R3/S3 in weekly trend direction
            if bullish_weekly and close[i] <= s3_aligned[i]:
                # Long at S3 in weekly uptrend
                signals[i] = 0.25
                position = 1
            elif bearish_weekly and close[i] >= r3_aligned[i]:
                # Short at R3 in weekly downtrend
                signals[i] = -0.25
                position = -1
            # Trend continuation breakout at R4/S4
            elif bullish_weekly and close[i] >= r4_aligned[i]:
                # Long breakout above R4 in weekly uptrend
                signals[i] = 0.25
                position = 1
            elif bearish_weekly and close[i] <= s4_aligned[i]:
                # Short breakdown below S4 in weekly downtrend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend turns bearish OR price re-enters mean-reversion zone
            if bearish_weekly or close[i] >= r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend turns bullish OR price re-enters mean-reversion zone
            if bullish_weekly or close[i] <= s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals