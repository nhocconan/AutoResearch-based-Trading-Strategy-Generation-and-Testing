#!/usr/bin/env python3
# 6h_12h_1d_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: 6s Camarilla R3/S3 breakout with 1d trend filter (EMA34) and volume spike.
# Camarilla levels from 1d provide institutional-grade support/resistance.
# Breakouts at R3/S3 with trend alignment and volume confirmation capture
# institutional breakouts while avoiding fakeouts. Works in bull/bear by requiring
# trend alignment, avoiding counter-trend traps. Targets 50-150 total trades over 4 years.

name = "6h_12h_1d_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels (using previous day's OHLC)
    # Camarilla: Close ± (High-Low) * multiplier
    # R3 = Close + (High-Low) * 1.1/2
    # S3 = Close - (High-Low) * 1.1/2
    # R4 = Close + (High-Low) * 1.1
    # S4 = Close - (High-Low) * 1.1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    camarilla_r4 = np.full(len(df_1d), np.nan)
    camarilla_s4 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):  # Start from 1 to use previous day's data
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        diff = prev_high - prev_low
        
        camarilla_r3[i] = prev_close + diff * 1.1 / 2
        camarilla_s3[i] = prev_close - diff * 1.1 / 2
        camarilla_r4[i] = prev_close + diff * 1.1
        camarilla_s4[i] = prev_close - diff * 1.1
    
    # Align Camarilla levels to 6h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average (4-period for 6h = 1 day)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for Camarilla (1d data) + EMA34 + vol MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: 1d close > EMA34
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        uptrend = close_1d_aligned[i] > ema_34_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation (2x average for significance)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Breakout above Camarilla R3 in uptrend with volume spike
            if close[i] > camarilla_r3_aligned[i] and uptrend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Camarilla S3 in downtrend with volume spike
            elif close[i] < camarilla_s3_aligned[i] and downtrend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: close below Camarilla R3 or trend fails
                if close[i] < camarilla_r3_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: close above Camarilla S3 or trend fails
                if close[i] > camarilla_s3_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals