#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1wTrend_Volume
Hypothesis: Use weekly Camarilla R3/S3 levels from the weekly pivot for breakout signals, with weekly EMA34 trend filter and volume confirmation. Camarilla levels work well in ranging markets (breakouts from R3/S3) and trending markets (pullbacks to R3/S3). Weekly trend filter ensures we only trade in the direction of the higher timeframe trend. Volume confirmation reduces false breakouts. Designed for 12h timeframe to limit trades (target: 50-150 over 4 years) and avoid fee drag while capturing significant moves.
"""

name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume"
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
    
    # Get weekly data for Camarilla calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (based on previous week's OHLC)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We use R3 and S3 as key levels
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    # Calculate Camarilla R3 and S3 for each week
    camarilla_r3 = wk_close + ((wk_high - wk_low) * 1.1 / 4)
    camarilla_s3 = wk_close - ((wk_high - wk_low) * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe (no extra delay needed as these are weekly levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Weekly EMA34 for trend filter
    wk_ema34 = pd.Series(wk_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    wk_ema34_aligned = align_htf_to_ltf(prices, df_1w, wk_ema34)
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(wk_ema34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly R3 with volume confirmation and above weekly EMA34
            if close[i] > camarilla_r3_aligned[i] and volume[i] > vol_ma[i] and close[i] > wk_ema34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly S3 with volume confirmation and below weekly EMA34
            elif close[i] < camarilla_s3_aligned[i] and volume[i] > vol_ma[i] and close[i] < wk_ema34_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly S3 or volume drops significantly
            if close[i] < camarilla_s3_aligned[i] or volume[i] < vol_ma[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly R3 or volume drops significantly
            if close[i] > camarilla_r3_aligned[i] or volume[i] < vol_ma[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals