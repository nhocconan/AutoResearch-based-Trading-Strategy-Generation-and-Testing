#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
# Long when price breaks above Camarilla R3 and close > 1d EMA34 with volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S3 and close < 1d EMA34 with volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to target 75-200 total trades over 4 years on 4h timeframe.
# Camarilla levels provide intraday structure; 1d EMA34 filters counter-trend noise; volume confirms momentum.
# Designed for fewer, higher-quality trades to avoid fee drag while working in both bull and bear markets.

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike"
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate prior day's OHLC for Camarilla levels
    df_1d_full = get_htf_data(prices, '1d')
    if len(df_1d_full) < 2:
        return np.zeros(n)
    # Use previous day's H, L, C for today's Camarilla levels
    high_1d = df_1d_full['high'].values
    low_1d = df_1d_full['low'].values
    close_1d = df_1d_full['close'].values
    
    # Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    # Align to 4h: each Camarilla level lasts for the entire 4h bars of the day
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d_full, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d_full, camarilla_s3)
    
    # Calculate average volume for confirmation (20-period)
    lookback = 20
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 1d EMA34, volume spike
            if (high[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 1d EMA34, volume spike
            elif (low[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 OR volume drops below average
            if (low[i] < camarilla_s3_aligned[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 OR volume drops below average
            if (high[i] > camarilla_r3_aligned[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals