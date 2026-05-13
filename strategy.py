#!/usr/bin/env python3
# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above R3 with volume > 1.5x average AND price > 1w EMA50.
# Short when price breaks below S3 with volume > 1.5x average AND price < 1w EMA50.
# Exit on opposite Camarilla level (R3/S3) or trend reversal.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 7-25 trades/year.
# Camarilla pivots from 1d provide intraday structure; 1w EMA50 filters for higher-timeframe trend.
# Volume confirmation reduces false breakouts. Works in bull markets via breakout continuation
# and in bear markets via faded rallies at resistance. 1d timeframe keeps trade frequency low.

name = "1d_Camarilla_R3S3_1wTrend_Volume_v1"
timeframe = "1d"
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
    
    # Get 1d data for Camarilla pivot calculation (already 1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels on 1d: R3, S3
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Using previous day's high/low/close to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First value will be NaN due to roll, handled by min_periods equivalent
    diff = prev_high - prev_low
    R3 = prev_close + 1.1 * diff / 2
    S3 = prev_close - 1.1 * diff / 2
    # Set first element to NaN (no previous day)
    R3[0] = np.nan
    S3[0] = np.nan
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for EMA50
        # Skip if any required data is NaN
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 with volume confirmation AND price > 1w EMA50
            if close[i] > R3[i] and volume_filter[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S3 with volume confirmation AND price < 1w EMA50
            elif close[i] < S3[i] and volume_filter[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S3 OR trend reversal (price < 1w EMA50)
            if close[i] < S3[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R3 OR trend reversal (price > 1w EMA50)
            if close[i] > R3[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals