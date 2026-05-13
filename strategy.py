#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation.
# Long when price breaks above R3 (1d) AND close > 1d EMA34 AND volume > 1.5x 20-period average.
# Short when price breaks below S3 (1d) AND close < 1d EMA34 AND volume > 1.5x 20-period average.
# Exit when price reverts to the 1d VWAP (mean reversion) OR trend filter fails.
# Uses 12h timeframe for lower frequency, Camarilla levels from 1d for structure,
# 1d EMA34 for trend filter, and volume for confirmation. Target: 50-150 total trades over 4 years (12-37/year).
# Works in bull via breakout continuation, bear via faded rallies and mean reversion to VWAP.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume_v1"
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
    
    # Get 12h data for price action
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Get 1d data for Camarilla levels, EMA34 trend filter, and VWAP exit
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    cam_high_low = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * cam_high_low
    camarilla_s3 = close_1d - 1.1 * cam_high_low
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate VWAP on 1d for exit signal (typical price * volume cumsum / volume cumsum)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_1d = (typical_price_1d * volume_1d).cumsum() / volume_1d.cumsum()
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume filter: current 12h volume > 1.5x 20-period average
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_filter_12h = volume_12h > (1.5 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or
            np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 AND trend filter bullish AND volume confirmation
            if close_12h[i] > camarilla_r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_filter_12h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND trend filter bearish AND volume confirmation
            elif close_12h[i] < camarilla_s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_filter_12h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to VWAP (mean reversion) OR trend filter fails
            if close[i] <= vwap_1d_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reverts to VWAP (mean reversion) OR trend filter fails
            if close[i] >= vwap_1d_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals