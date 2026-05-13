#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike (>2.0x 20-bar avg) for confirmation.
# Uses 1d EMA34 for trend alignment (HTF), 4h Camarilla R3/S3 levels for breakout entry, and volume confirmation to avoid false breakouts.
# Designed for low-to-moderate trade frequency (target 75-200 total over 4 years) to minimize fee drag while capturing strong trends.
# Works in both bull and bear markets by following the 1d trend direction and requiring volume confirmation.
# Camarilla levels provide structured support/resistance; breakouts with volume indicate institutional participation.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_v1"
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels (based on previous day's OHLC)
    # For 4h timeframe, we use daily OHLC to calculate Camarilla levels
    # We'll use the previous completed day's OHLC
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day, then align to 4h
    high_1d = df_1d_ohlc['high'].values
    low_1d = df_1d_ohlc['low'].values
    close_1d_ohlc = df_1d_ohlc['close'].values
    
    # Camarilla R3 and S3 levels
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    camarilla_r3 = close_1d_ohlc + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d_ohlc - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 4h timeframe (using previous day's levels for current 4h bars)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_s3)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback for volume avg
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 1d EMA34, volume spike (>2.0x avg)
            if (high[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 1d EMA34, volume spike (>2.0x avg)
            elif (low[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price breaks below Camarilla S3 or volume drops significantly
            if (low[i] < camarilla_s3_aligned[i]) or (volume[i] < 0.4 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if price breaks above Camarilla R3 or volume drops significantly
            if (high[i] > camarilla_r3_aligned[i]) or (volume[i] < 0.4 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals