#!/usr/bin/env python3
# 12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_FILTER
# Hypothesis: Camarilla pivot levels (R3, S3) from daily timeframe act as strong support/resistance.
# Price breaking above R3 with bullish daily trend (close > EMA34) signals strong upward momentum.
# Price breaking below S3 with bearish daily trend (close < EMA34) signals strong downward momentum.
# Works in both bull and bear markets by following institutional price levels and trend.
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years).

name = "12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_FILTER"
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
    
    # Daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla pivot levels
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    cam_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    cam_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # EMA34 for trend filter
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 12h timeframe
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one day of data
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]) or 
            np.isnan(ema34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 in uptrend
            if (close[i] > cam_r3_aligned[i] and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 in downtrend
            elif (close[i] < cam_s3_aligned[i] and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below R3 or trend reversal
            if (close[i] < cam_r3_aligned[i] or 
                close[i] < ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above S3 or trend reversal
            if (close[i] > cam_s3_aligned[i] or 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals