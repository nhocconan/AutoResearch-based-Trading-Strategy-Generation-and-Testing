#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 with volume > 1.3x average AND price > 1d EMA34.
# Short when price breaks below Camarilla S3 with volume > 1.3x average AND price < 1d EMA34.
# Exit on opposite Camarilla level (S3 for long, R3 for short) or trend reversal.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
# Camarilla levels provide institutional support/resistance that work in ranging markets,
# while EMA34 filter ensures we only trade with the higher timeframe trend, reducing false breakouts.
# Volume confirmation adds conviction to breakouts.

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
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for 12h timeframe
    # Pivot point = (high + low + close) / 3
    # Range = high - low
    # R3 = close + (range * 1.1/2)
    # S3 = close - (range * 1.1/2)
    typical_price = (high_12h + low_12h + close_12h) / 3
    price_range = high_12h - low_12h
    r3_12h = close_12h + (price_range * 1.1 / 2)
    s3_12h = close_12h - (price_range * 1.1 / 2)
    
    # Since we're using 12h data for 12h timeframe, no additional alignment needed
    r3_12h_aligned = r3_12h
    s3_12h_aligned = s3_12h
    
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
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Camarilla R3 with volume confirmation AND price > 1d EMA34
            if close[i] > r3_12h_aligned[i] and volume_filter[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Camarilla S3 with volume confirmation AND price < 1d EMA34
            elif close[i] < s3_12h_aligned[i] and volume_filter[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Camarilla S3 OR trend reversal (price < 1d EMA34)
            if close[i] < s3_12h_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Camarilla R3 OR trend reversal (price > 1d EMA34)
            if close[i] > r3_12h_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals