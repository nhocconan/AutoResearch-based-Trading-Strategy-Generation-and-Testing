#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1w trend filter and volume confirmation.
# Long when price breaks above R3 AND close > 1w EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below S3 AND close < 1w EMA50 AND volume > 1.5x 20-period average
# Exit when price reverts to the Camarilla pivot (PP) level or trend reverses
# Uses 12h timeframe for lower frequency (~12-37 trades/year), Camarilla for structure,
# 1w EMA for trend filter, volume for confirmation. Works in bull via breakout continuation,
# bear via faded rallies and short breakdowns.

name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume_v1"
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
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Camarilla levels for 12h
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We only need R3 and S3 for breakout, PP for exit
    range_12h = high_12h - low_12h
    r3_12h = close_12h + 1.1 * range_12h
    s3_12h = close_12h - 1.1 * range_12h
    pp_12h = (high_12h + low_12h + close_12h) / 3.0  # Pivot Point
    
    # Volume filter: current 12h volume > 1.5x 20-period average
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_filter_12h = volume_12h > (1.5 * vol_ma_12h)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(pp_12h[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 AND price > 1w EMA50 AND volume confirmation
            if close[i] > r3_12h[i] and close[i] > ema50_1w_aligned[i] and volume_filter_12h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND price < 1w EMA50 AND volume confirmation
            elif close[i] < s3_12h[i] and close[i] < ema50_1w_aligned[i] and volume_filter_12h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to PP (mean reversion) OR trend reversal (price < 1w EMA50)
            if close[i] <= pp_12h[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reverts to PP (mean reversion) OR trend reversal (price > 1w EMA50)
            if close[i] >= pp_12h[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals