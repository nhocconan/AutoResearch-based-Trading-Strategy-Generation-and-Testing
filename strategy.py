#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter and volume spike confirmation.
# Long when price breaks above R3 (camarilla resistance) with volume > 2x average AND price > 1d EMA50
# Short when price breaks below S3 (camarilla support) with volume > 2x average AND price < 1d EMA50
# Exit when price reverts to Camarilla pivot point (PP) or trend reverses (price crosses 1d EMA50 opposite direction)
# Uses 6h timeframe for lower frequency, Camarilla from prior 1d for structure, volume for confirmation, 1d EMA for trend filter.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via breakout continuation, bear via faded rallies at structure.

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
    
    # Get 1d data for Camarilla pivot calculation (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for prior 1d
    # PP = (high + low + close) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # R3 = PP + (high - low) * 1.1 / 2
    r3_1d = pp_1d + (high_1d - low_1d) * 1.1 / 2.0
    # S3 = PP - (high - low) * 1.1 / 2
    s3_1d = pp_1d - (high_1d - low_1d) * 1.1 / 2.0
    # R4 = PP + (high - low) * 1.1
    r4_1d = pp_1d + (high_1d - low_1d) * 1.1
    # S4 = PP - (high - low) * 1.1
    s4_1d = pp_1d - (high_1d - low_1d) * 1.1
    
    # Align Camarilla levels to 6h timeframe (wait for 1d bar to close)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Get 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 6h volume > 2.0x 20-period average
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with volume spike AND price > 1d EMA50 (uptrend)
            if close[i] > r3_1d_aligned[i] and volume_filter[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume spike AND price < 1d EMA50 (downtrend)
            elif close[i] < s3_1d_aligned[i] and volume_filter[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to PP (mean reversion) OR trend reversal (price < 1d EMA50)
            if close[i] <= pp_1d_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reverts to PP (mean reversion) OR trend reversal (price > 1d EMA50)
            if close[i] >= pp_1d_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals