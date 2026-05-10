#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla pivot levels (R3, S3) from daily timeframe act as strong support/resistance.
Breakout above R3 or below S3 with volume confirmation and aligned 1d trend (price > EMA34 for longs,
price < EMA34 for shorts) captures institutional breakout moves. Works in both bull and bear markets
by trading breakouts in the direction of the higher timeframe trend. Uses 12h timeframe to limit
trades to 12-37/year, reducing fee drag. Volume spike (>2x 20-period average) confirms institutional
participation. Target: 50-150 total trades over 4 years.
"""

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
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
    
    # Daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    # R3 = Close + 1.1 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe (available after daily bar closes)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA34 for trend filter
    if len(close_1d) >= 34:
        ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False).mean().values
    else:
        ema34_1d = np.full_like(close_1d, np.nan)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 20-period volume average for volume confirmation
    if len(df_1d) >= 20:
        vol_ma20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    else:
        vol_ma20_1d = np.full_like(df_1d['volume'].values, np.nan)
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have all indicators ready
    start_idx = 20  # Need 20 days for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema34_aligned[i]) or np.isnan(vol_ma20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 2x 20-day average volume (scaled to 12h)
        # 1 day = 2 periods of 12h, so scale daily volume by 1/2
        vol_12h_equiv = vol_ma20_aligned[i] / 2.0
        volume_confirm = volume[i] > 2.0 * vol_12h_equiv
        
        # Trend filter: price vs 1d EMA34
        is_uptrend = close[i] > ema34_aligned[i]
        is_downtrend = close[i] < ema34_aligned[i]
        
        if position == 0:
            # Long: break above R3 with volume and uptrend
            if close[i] > r3_aligned[i] and volume_confirm and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume and downtrend
            elif close[i] < s3_aligned[i] and volume_confirm and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns below R3 or trend breaks
            if close[i] < r3_aligned[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above S3 or trend breaks
            if close[i] > s3_aligned[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals