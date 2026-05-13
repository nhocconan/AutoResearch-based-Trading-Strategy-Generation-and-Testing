#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter and volume spike confirmation.
# Camarilla pivot levels calculated from 1d OHLC: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
# Long when price breaks above R3 with volume > 2x 20-period average AND price > 1d EMA50
# Short when price breaks below S3 with volume > 2x 20-period average AND price < 1d EMA50
# Exit when price returns to 1d VWAP (mean reversion to daily fair value) OR opposite Camarilla level touched
# Uses 6h timeframe for lower frequency, Camarilla for structure, 1d EMA for trend, volume spike for conviction.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via breakout continuation, bear via faded rallies to VWAP.

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1"
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
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 levels from 1d OHLC
    # R3 = close + 1.1*(high-low)/2
    # S3 = close - 1.1*(high-low)/2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 6h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d VWAP for exit (mean reversion target)
    # VWAP = sum(price * volume) / sum(volume) for the day
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_1d = (np.cumsum(typical_price_1d * df_1d['volume'].values) / 
               np.cumsum(df_1d['volume'].values))
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume filter: current 6h volume > 2x 20-period average
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter_6h = volume > (2.0 * vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or
            np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with volume spike AND above 1d EMA50 (uptrend)
            if close[i] > camarilla_r3_aligned[i] and volume_filter_6h[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume spike AND below 1d EMA50 (downtrend)
            elif close[i] < camarilla_s3_aligned[i] and volume_filter_6h[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to 1d VWAP (mean reversion) OR touches S3 (contrarian level)
            if close[i] <= vwap_1d_aligned[i] or close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to 1d VWAP (mean reversion) OR touches R3 (contrarian level)
            if close[i] >= vwap_1d_aligned[i] or close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals