#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above R3 with volume > 1.3x average AND price > 1d EMA34.
# Short when price breaks below S3 with volume > 1.3x average AND price < 1d EMA34.
# Exit on opposite Camarilla level (S3 for longs, R3 for shorts) or trend reversal.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
# Camarilla levels from 1d provide institutional support/resistance that works in both bull (breakout continuation) and bear (faded rallies at resistance) markets.
# 12h timeframe reduces trade frequency vs lower TFs, improving fee drag profile.

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
    
    # Get 12h data for Camarilla level calculation (using 12h OHLC)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels on 12h: based on previous 12h bar's range
    # R3 = close + 1.1*(high - low), S3 = close - 1.1*(high - low)
    # We use the previous completed 12h bar to avoid look-ahead
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h[0] = np.nan  # First bar has no previous
    prev_high_12h[0] = np.nan
    prev_low_12h[0] = np.nan
    
    R3_12h = prev_close_12h + 1.1 * (prev_high_12h - prev_low_12h)
    S3_12h = prev_close_12h - 1.1 * (prev_high_12h - prev_low_12h)
    
    # Since we're using 12h data for 12h timeframe, no additional alignment needed
    R3_12h_aligned = R3_12h
    S3_12h_aligned = S3_12h
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(R3_12h_aligned[i]) or np.isnan(S3_12h_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 with volume confirmation AND price > 1d EMA34
            if close[i] > R3_12h_aligned[i] and volume_filter[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S3 with volume confirmation AND price < 1d EMA34
            elif close[i] < S3_12h_aligned[i] and volume_filter[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S3 OR trend reversal (price < 1d EMA34)
            if close[i] < S3_12h_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R3 OR trend reversal (price > 1d EMA34)
            if close[i] > R3_12h_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals