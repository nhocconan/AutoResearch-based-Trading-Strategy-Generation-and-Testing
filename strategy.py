#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_S1_S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels from daily timeframe provide high-probability support/resistance. 
Breakout above S3 (strong resistance) or below S1 (strong support) with volume confirmation and 
daily EMA trend filter captures momentum moves. Works in bull markets (breakouts continue) and 
bear markets (breakdowns continue). Designed for 4h timeframe with tight entry conditions 
(~25-40 trades/year) to avoid fee drag.
"""

name = "4h_Camarilla_Pivot_S1_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each day
    # Based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: S1 = C - (H-L)*1.12/6, S3 = C - (H-L)*1.12/2
    # R1 = C + (H-L)*1.12/6, R3 = C + (H-L)*1.12/2
    range_1d = high_1d - low_1d
    s1_1d = close_1d - (range_1d * 1.12 / 6)
    s3_1d = close_1d - (range_1d * 1.12 / 2)
    r1_1d = close_1d + (range_1d * 1.12 / 6)
    r3_1d = close_1d + (range_1d * 1.12 / 2)
    
    # Align daily levels to 4h timeframe (wait for daily close)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # Get daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike (current volume > 2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2x 20-period average
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Break above R3 with volume spike and price above daily EMA34 (bullish trend)
            if close[i] > r3_aligned[i] and vol_spike and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume spike and price below daily EMA34 (bearish trend)
            elif close[i] < s3_aligned[i] and vol_spike and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below R1 or daily EMA34
            if close[i] < r1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above S1 or daily EMA34
            if close[i] > s1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals