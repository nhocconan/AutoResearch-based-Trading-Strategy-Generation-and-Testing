#!/usr/bin/env python3
"""
1h_Camarilla_Breakout_4hTrend_1dRegime_v1
Hypothesis: On 1h timeframe, using 4h EMA for trend direction and 1d Camarilla levels (R3/S3) for breakout entries with volume confirmation reduces whipsaws in ranging markets. The 1d regime filter (price vs 200 EMA) ensures we only take breakouts in the direction of the higher timeframe trend, improving win rate in both bull and bear markets. Session filter (08-20 UTC) reduces noise trades. Target: 60-150 total trades over 4 years (15-37/year).
"""

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
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h EMA for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d Camarilla levels (R3, S3) - more significant breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    high_1d_prev = np.concatenate([[np.nan], high_1d[:-1]])
    low_1d_prev = np.concatenate([[np.nan], low_1d[:-1]])
    close_1d_prev = np.concatenate([[np.nan], close_1d[:-1]])
    
    camarilla_range = high_1d_prev - low_1d_prev
    r3 = close_1d_prev + 1.1 * camarilla_range / 4
    s3 = close_1d_prev - 1.1 * camarilla_range / 4
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA200 for regime filter (bull/bear)
    ema_200 = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # 1h volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20, 200)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(ema_200_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # 4h trend filter
        uptrend_4h = close[i] > ema_4h_aligned[i]
        downtrend_4h = close[i] < ema_4h_aligned[i]
        
        # 1d regime filter: price above/below 200 EMA
        bull_regime = close[i] > ema_200_aligned[i]
        bear_regime = close[i] < ema_200_aligned[i]
        
        # Volume confirmation
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Camarilla breakout conditions (R3/S3)
        breakout_r3 = close[i] > r3_aligned[i]
        breakout_s3 = close[i] < s3_aligned[i]
        
        # Long logic: breakout above R3 in 4h uptrend + bull regime + volume
        if uptrend_4h and bull_regime and volume_spike and breakout_r3:
            if position != 1:
                signals[i] = 0.20
                position = 1
            else:
                signals[i] = 0.20
        # Short logic: breakout below S3 in 4h downtrend + bear regime + volume
        elif downtrend_4h and bear_regime and volume_spike and breakout_s3:
            if position != -1:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = -0.20
        # Exit conditions: loss of 4h trend or regime change
        elif position == 1 and (not uptrend_4h or not bull_regime):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not downtrend_4h or not bear_regime):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_Breakout_4hTrend_1dRegime_v1"
timeframe = "1h"
leverage = 1.0