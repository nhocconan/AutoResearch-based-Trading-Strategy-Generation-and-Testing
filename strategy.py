#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above R3 with volume > 1.4x average AND price > 1d EMA50.
# Short when price breaks below S3 with volume > 1.4x average AND price < 1d EMA50.
# Exit on opposite level (S3 for long, R3 for short) or trend reversal.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
# Camarilla pivots from 1d provide strong intraday support/resistance levels that work in ranging and trending markets.
# Volume confirmation reduces false breakouts. EMA50 filter ensures trades align with higher timeframe trend.
# 12h timeframe keeps trade frequency low to overcome fee drag in bear markets like 2025.

name = "12h_Camarilla_R3S3_1dTrend_Volume_v1"
timeframe = "12h"
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d: R3, S3
    # R3 = close + 1.1*(high-low)*1.1/4 = close + 1.1*(high-low)*0.275
    # S3 = close - 1.1*(high-low)*1.1/4 = close - 1.1*(high-low)*0.275
    # Actually standard Camarilla: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    # But many use: R4 = close + (high-low)*1.1/2, R3 = close + (high-low)*1.1/4
    # We'll use: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    rangep = high_1d - low_1d
    r3_1d = close_1d + (rangep * 1.1 / 4)
    s3_1d = close_1d - (rangep * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Get 1d data for EMA50 trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.4x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.4 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 with volume confirmation AND price > 1d EMA50
            if close[i] > r3_1d_aligned[i] and volume_filter[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S3 with volume confirmation AND price < 1d EMA50
            elif close[i] < s3_1d_aligned[i] and volume_filter[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S3 OR trend reversal (price < 1d EMA50)
            if close[i] < s3_1d_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R3 OR trend reversal (price > 1d EMA50)
            if close[i] > r3_1d_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals