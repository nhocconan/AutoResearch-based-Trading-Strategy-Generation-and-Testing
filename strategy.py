#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 breakout with 1w trend filter and volume confirmation.
# Long when price breaks above R3 AND 1w EMA50 is rising AND volume > 1.5x average
# Short when price breaks below S3 AND 1w EMA50 is falling AND volume > 1.5x average
# Exit when price reverts to the 1d VWAP or trend reverses
# Uses 6h timeframe for lower frequency, Camarilla levels from 1d for structure, 1w EMA for trend filter, volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via breakout continuation, bear via faded rallies.

name = "6h_Camarilla_R3S3_Breakout_1wTrend_Volume_v1"
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
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for current 1d bar (based on previous 1d bar's range)
    # R3 = close + 1.1*(high - low)/2, S3 = close - 1.1*(high - low)/2
    # Using previous 1d bar to avoid look-ahead
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_range = prev_high_1d - prev_low_1d
    r3 = prev_close_1d + 1.1 * camarilla_range / 2
    s3 = prev_close_1d - 1.1 * camarilla_range / 2
    
    # Al Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current 6h volume > 1.5x 20-period average
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter_6h = volume > (1.5 * vol_ma_6h)
    
    # Get 1d data for VWAP exit
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_1d = (typical_price_1d * volume_1d).cumsum() / volume_1d.cumsum()
    # Handle first bar
    vwap_1d = np.where(np.isclose(volume_1d.cumsum(), 0), typical_price_1d, vwap_1d)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_6h[i]) or
            np.isnan(vwap_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 AND 1w EMA50 is rising AND volume confirmation
            if close[i] > r3_aligned[i] and ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and volume_filter_6h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND 1w EMA50 is falling AND volume confirmation
            elif close[i] < s3_aligned[i] and ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and volume_filter_6h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to 1d VWAP OR trend reversal (1w EMA50 falling)
            if close[i] < vwap_1d_aligned[i] or ema50_1w_aligned[i] < ema50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reverts to 1d VWAP OR trend reversal (1w EMA50 rising)
            if close[i] > vwap_1d_aligned[i] or ema50_1w_aligned[i] > ema50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals