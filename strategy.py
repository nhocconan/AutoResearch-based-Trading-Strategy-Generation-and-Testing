#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1w trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 AND 1w close > 1w EMA50 AND volume > 1.5x average.
# Short when price breaks below Camarilla S3 AND 1w close < 1w EMA50 AND volume > 1.5x average.
# Exit when price returns to Camarilla Pivot point (mean reversion to equilibrium).
# Uses 12h timeframe for lower frequency, Camarilla levels from 1d for structure, 1w EMA for trend filter.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via breakout continuation, bear via faded rallies to pivot.

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
    
    # Get 1d data for Camarilla pivot calculation (yesterday's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for today (based on yesterday's 1d candle)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use R3 and S3 for breakout, Pivot (close) for exit
    diff_1d = high_1d - low_1d
    camarilla_pivot = close_1d  # Pivot = previous close
    camarilla_r3 = close_1d + 1.1 * diff_1d
    camarilla_s3 = close_1d - 1.1 * diff_1d
    
    # Align Camarilla levels to 12h timeframe (yesterday's levels available after 1d close)
    pivot_12h = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: current 12h volume > 1.5x 20-period average
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_12h)
    
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
        if (np.isnan(pivot_12h[i]) or np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 AND 1w close > 1w EMA50 AND volume confirmation
            if close[i] > r3_12h[i] and close_1w[i] > ema50_1w[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S3 AND 1w close < 1w EMA50 AND volume confirmation
            elif close[i] < s3_12h[i] and close_1w[i] < ema50_1w[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to pivot point (mean reversion)
            if close[i] <= pivot_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to pivot point (mean reversion)
            if close[i] >= pivot_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals