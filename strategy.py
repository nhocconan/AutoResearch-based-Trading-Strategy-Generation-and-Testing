#!/usr/bin/env python3

# 4H_CAMARILLA_R3_S3_BREAKOUT_12H_EMA50_TREND_VOLUME
# Hypothesis: Camarilla R3/S3 levels derived from 12h candles represent strong intraday support/resistance.
# Price breaking above R3 with volume and 12h EMA50 uptrend signals continuation long.
# Price breaking below S3 with volume and 12h EMA50 downtrend signals continuation short.
# Works in bull (buy breakouts) and bear (sell breakdowns) markets by following 12h trend.
# Target: 25-40 trades/year on 4h timeframe (~100-160 total over 4 years).

name = "4H_CAMARILLA_R3_S3_BREAKOUT_12H_EMA50_TREND_VOLUME"
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
    
    # 12h data for Camarilla calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla R3 and S3 levels from previous 12h bar
    camarilla_r3 = np.full(len(close_12h), np.nan)
    camarilla_s3 = np.full(len(close_12h), np.nan)
    
    for i in range(1, len(close_12h)):
        ph = high_12h[i-1]
        pl = low_12h[i-1]
        pc = close_12h[i-1]
        range_val = ph - pl
        
        camarilla_r3[i] = pc + range_val * 1.1 / 4
        camarilla_s3[i] = pc - range_val * 1.1 / 4
    
    # EMA50 for 12h trend filter
    ema50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: current 4h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > 1.5 * vol_ma
    
    # Align all 12h data to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any critical data is not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 with volume spike in uptrend
            if (high[i] > camarilla_r3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume spike in downtrend
            elif (low[i] < camarilla_s3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below R3 or trend reversal
            if (close[i] < camarilla_r3_aligned[i] or 
                close[i] < ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above S3 or trend reversal
            if (close[i] > camarilla_s3_aligned[i] or 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals