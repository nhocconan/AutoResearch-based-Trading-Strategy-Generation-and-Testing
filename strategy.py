#!/usr/bin/env python3
name = "1d_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    trend_up = close > ema34_1w_aligned
    trend_down = close < ema34_1w_aligned
    
    # Daily Camarilla levels (based on previous day)
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = high[0]
    low_prev[0] = low[0]
    close_prev[0] = close[0]
    
    range_prev = high_prev - low_prev
    R3 = close_prev + range_prev * 1.1000
    S3 = close_prev - range_prev * 1.1000
    
    # Volume filter: today's volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 days for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(volume_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout above R3 with volume and 1w uptrend
            if close[i] > R3[i] and volume_filter[i] and trend_up[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown below S3 with volume and 1w downtrend
            elif close[i] < S3[i] and volume_filter[i] and trend_down[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price retests S3 or trend turns down
            if close[i] < S3[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price retests R3 or trend turns up
            if close[i] > R3[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakouts with 1-week trend filter and volume confirmation
# captures institutional breakout moves in both bull and bear markets.
# Long when price breaks above R3 with volume in 1w uptrend, short when breaks below S3 with volume in 1w downtrend.
# Camarilla levels provide mathematically derived support/resistance based on prior day's range.
# Volume filter ensures breakouts have institutional participation.
# 1w trend filter avoids counter-trend whipsaws.
# Position size 0.25 manages risk through volatility regimes.