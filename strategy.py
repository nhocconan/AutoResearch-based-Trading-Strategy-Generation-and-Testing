#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike (>2.0x 20-bar avg) for confirmation.
# Uses 1d EMA34 for trend alignment (HTF), 6h Camarilla levels (R3/S3) for breakout entry, and volume confirmation to avoid false breakouts.
# Designed for low trade frequency (target 50-150 total over 4 years) to minimize fee drag while capturing strong trends in both bull and bear markets.
# Works in both bull and bear markets by following the 1d trend direction and requiring volume confirmation.

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels (based on previous day's OHLC)
    # We use daily OHLC from 1d timeframe to compute Camarilla for 6h bars
    # Camarilla R3 = close + 1.1*(high-low)*1.1/4
    # Camarilla S3 = close - 1.1*(high-low)*1.1/4
    # Actually: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    # But standard Camarilla: R4 = close + (high-low)*1.1/2, R3 = close + (high-low)*1.1/4
    # We'll compute from 1d OHLC and align to 6h
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe (wait for 1d bar to close)
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_1d_aligned[i]) or 
            np.isnan(camarilla_s3_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 1d EMA34, volume spike (>2.0x avg)
            if (high[i] > camarilla_r3_1d_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 1d EMA34, volume spike (>2.0x avg)
            elif (low[i] < camarilla_s3_1d_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price breaks below Camarilla S3 or volume drops
            if (low[i] < camarilla_s3_1d_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if price breaks above Camarilla R3 or volume drops
            if (high[i] > camarilla_r3_1d_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals