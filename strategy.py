#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Uses Camarilla pivot levels from 4h for structure, 12h EMA50 for trend filter (proven BTC/ETH edge),
# and volume spike for confirmation. Designed for 20-30 trades/year to minimize fee drag.
# Works in bull markets via upside breakouts at R3 and in bear markets via downside breakdowns at S3.
# The 12h EMA50 provides a smooth trend filter that adapts to changing regimes while avoiding whipsaw.

name = "4h_Camarilla_R3S3_12hEMA50_VolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot levels (use same timeframe for structure)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior completed 4h bar
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    camarilla_r3 = close_4h + (high_4h - low_4h) * 1.1 / 4
    camarilla_s3 = close_4h - (high_4h - low_4h) * 1.1 / 4
    
    camarilla_r3_shifted = np.roll(camarilla_r3, 1)
    camarilla_s3_shifted = np.roll(camarilla_s3, 1)
    camarilla_r3_shifted[0] = np.nan
    camarilla_s3_shifted[0] = np.nan
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_shifted)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_shifted)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 trend filter from prior completed 12h bar
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_shifted = np.roll(ema50_12h, 1)
    ema50_12h_shifted[0] = np.nan
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Camarilla R3 AND 12h EMA50 uptrend AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema50_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Camarilla S3 AND 12h EMA50 downtrend AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema50_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S3 OR below 12h EMA50
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla R3 OR above 12h EMA50
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals