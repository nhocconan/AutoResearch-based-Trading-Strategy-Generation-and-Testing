#!/usr/bin/env python3
# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 level with volume > 1.5x average AND price > 1w EMA34.
# Short when price breaks below Camarilla S3 level with volume > 1.5x average AND price < 1w EMA34.
# Exit on opposite Camarilla level (S3 for long, R3 for short) or trend reversal.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 7-25 trades/year.
# Works in bull markets via breakout continuation and in bear markets via faded rallies at resistance.
# 1d timeframe reduces trade frequency vs lower TFs, improving fee drag profile.

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
    
    # Get 1d data for Camarilla pivot calculation (using daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d: based on previous day's OHLC
    # Camarilla R3 = close + 1.1*(high - low)*1.1/2
    # Camarilla S3 = close - 1.1*(high - low)*1.1/2
    # Actually standard Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We'll use R3 and S3 as breakout levels
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # Set first value to avoid look-ahead (use same day's data for first bar)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 1d timeframe (already aligned as we used 1d data)
    camarilla_r3_aligned = camarilla_r3
    camarilla_s3_aligned = camarilla_s3
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on 1w close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Camarilla R3 with volume confirmation AND price > 1w EMA34
            if close[i] > camarilla_r3_aligned[i] and volume_filter[i] and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Camarilla S3 with volume confirmation AND price < 1w EMA34
            elif close[i] < camarilla_s3_aligned[i] and volume_filter[i] and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Camarilla S3 OR trend reversal (price < 1w EMA34)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Camarilla R3 OR trend reversal (price > 1w EMA34)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals