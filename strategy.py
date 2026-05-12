#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Uses Camarilla R3/S3 levels from daily timeframe for reversal entries,
# confirmed by daily trend (EMA50) and volume spike on 12h. Targets mean reversion
# in ranging markets and breakouts in trending markets. Works in both bull/bear by
# adapting to trend direction. Low trade frequency via strict confluence.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >2.0x 30-period average (on 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3) from previous day
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    # Using previous day's values to avoid look-ahead
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * range_1d
    camarilla_s3 = close_1d - 1.1 * range_1d
    
    # Shift by 1 to use previous day's levels
    camarilla_r3 = np.roll(camarilla_r3, 1)
    camarilla_s3 = np.roll(camarilla_s3, 1)
    camarilla_r3[0] = np.nan
    camarilla_s3[0] = np.nan
    
    # Daily trend filter: EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all daily data to 12h timeframe
    camarilla_r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(camarilla_r3_12h[i]) or
            np.isnan(camarilla_s3_12h[i]) or
            np.isnan(ema_50_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price crosses above S3 (support) in uptrend OR below R3 (resistance) in downtrend
            # Actually, we want mean reversion: buy near S3 in uptrend, sell near R3 in downtrend
            if close[i] > camarilla_s3_12h[i] and close[i] > ema_50_12h[i]:
                # Price above S3 and above EMA50 = long setup
                if volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
            elif close[i] < camarilla_r3_12h[i] and close[i] < ema_50_12h[i]:
                # Price below R3 and below EMA50 = short setup
                if volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S3 OR reaches R3 (take profit)
            if close[i] < camarilla_s3_12h[i] or close[i] > camarilla_r3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R3 OR reaches S3 (take profit)
            if close[i] > camarilla_r3_12h[i] or close[i] < camarilla_s3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals