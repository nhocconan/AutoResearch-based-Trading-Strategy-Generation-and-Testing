#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter and volume spike.
# Long when price breaks above Camarilla R3 AND 1d EMA34 uptrend AND volume > 2.0x 20-period average.
# Short when price breaks below Camarilla S3 AND 1d EMA34 downtrend AND volume > 2.0x 20-period average.
# Exit on opposite Camarilla level touch (S3 for long, R3 for short) or trend reversal.
# Uses 4h timeframe for optimal trade frequency, Camarilla for institutional levels, 1d EMA for trend, volume for confirmation.
# Target: 75-200 total trades over 4 years (19-50/year). Works in bull via breakout continuation, bear via faded rallies at key levels.

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_v1"
timeframe = "4h"
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
    open_ = prices['open'].values
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Camarilla levels for 4h
    # Camarilla: R4 = close + 1.5*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/2
    # S3 = close - 1.1*(high-low)*1.1/2, S4 = close - 1.5*(high-low)*1.1/2
    # Actually standard: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    # Using standard Camarilla: R3 = C + 1.1*(H-L), S3 = C - 1.1*(H-L)
    camarilla_range = high_4h - low_4h
    camarilla_r3 = close_4h + 1.1 * camarilla_range
    camarilla_s3 = close_4h - 1.1 * camarilla_range
    
    # Volume filter: current 4h volume > 2.0x 20-period average
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_filter_4h = volume_4h > (2.0 * vol_ma_4h)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 AND 1d EMA34 uptrend AND volume confirmation
            if close[i] > camarilla_r3[i] and close[i] > ema34_1d_aligned[i] and volume_filter_4h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND 1d EMA34 downtrend AND volume confirmation
            elif close[i] < camarilla_s3[i] and close[i] < ema34_1d_aligned[i] and volume_filter_4h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches S3 OR trend reversal (price < 1d EMA34)
            if close[i] < camarilla_s3[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches R3 OR trend reversal (price > 1d EMA34)
            if close[i] > camarilla_r3[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals