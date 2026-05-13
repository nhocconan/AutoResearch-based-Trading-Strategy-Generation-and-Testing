#!/usr/bin/env python3
"""
6h_WeeklyPivot_Breakout_Trend_Filter
Hypothesis: Weekly pivot levels act as strong support/resistance. Breakout above weekly R3 or below S3 with
daily trend alignment and volume confirmation captures institutional flow. Works in bull/bear by using
trend filters to avoid false breakouts in ranging markets. Target: 15-35 trades/year.
"""

name = "6h_WeeklyPivot_Breakout_Trend_Filter"
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
    volume = prices['volume'].values
    
    # Daily trend filter: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_daily = close > ema_50
    downtrend_daily = close < ema_50
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    # R3 = H + 2*(P-L), S3 = L - 2*(H-P)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly levels to 6h timeframe (wait for weekly close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        r3 = weekly_r3_aligned[i]
        s3 = weekly_s3_aligned[i]
        uptrend = uptrend_daily[i]
        downtrend = downtrend_daily[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above weekly R3, daily uptrend, volume confirmation
            if close[i] > r3 and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below weekly S3, daily downtrend, volume confirmation
            elif close[i] < s3 and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to weekly pivot or daily trend turns down
            if close[i] < weekly_pivot_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to weekly pivot or daily trend turns up
            if close[i] > weekly_pivot_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals