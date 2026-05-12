#!/usr/bin/env python3
"""
1D_WEEKLY_CAMARILLA_R3_S3_BREAKOUT_VOLUME
Hypothesis: Weekly Cambria R3/S3 breakout with daily volume spike confirmation.
Trades weekly breakouts on daily timeframe with volume confirmation to capture
institutional participation. Designed for low trade frequency (<25/year) to
minimize fee drag and work in both bull and bear markets.
"""
name = "1D_WEEKLY_CAMARILLA_R3_S3_BREAKOUT_VOLUME"
timeframe = "1d"
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
    
    # Calculate weekly Camarilla levels from previous week
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # We need previous week's OHLC
    weekly_high = np.full(n, np.nan)
    weekly_low = np.full(n, np.nan)
    weekly_close = np.full(n, np.nan)
    
    for i in range(1, n):
        weekly_high[i] = high[i-1]
        weekly_low[i] = low[i-1]
        weekly_close[i] = close[i-1]
    
    # Calculate weekly Camarilla levels
    weekly_range = weekly_high - weekly_low
    camarilla_r3 = weekly_close + weekly_range * 1.1 / 2
    camarilla_s3 = weekly_close - weekly_range * 1.1 / 2
    
    # Daily data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d / vol_ma_20d  # Current volume / 20-day average
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above weekly R3 with volume spike
            if (high[i] > camarilla_r3[i] and 
                vol_spike_aligned[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly S3 with volume spike
            elif (low[i] < camarilla_s3[i] and 
                  vol_spike_aligned[i] > 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below weekly S3 (reversion to mean)
            if close[i] < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above weekly R3 (reversion to mean)
            if close[i] > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals