#!/usr/bin/env python3
# 1H_CAMARILLA_PIVOT_BREAKOUT_4H_TREND_1D_VOLUME_FILTER
# Hypothesis: Use Camarilla pivot levels (R3/S3) on 4h for directional bias, breakout on 1h with volume confirmation.
# Only trade when 1d trend is aligned (price above/below EMA34). Reduces false breakouts in chop.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.
# Works in bull/bear by using 1d trend filter and volume confirmation to avoid whipsaws.

name = "1H_CAMARILLA_PIVOT_BREAKOUT_4H_TREND_1D_VOLUME_FILTER"
timeframe = "1h"
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
    
    # 4h data for Camarilla pivot levels (R3/S3)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar
    # Pivot = (H+L+C)/3
    # R3 = Pivot + 1.1*(H-L)
    # S3 = Pivot - 1.1*(H-L)
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    r3_4h = pivot_4h + 1.1 * (high_4h - low_4h)
    s3_4h = pivot_4h - 1.1 * (high_4h - low_4h)
    
    # 1d data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d volume moving average (20-period) for volume filter
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1h volume spike: current volume > 1.5 * 20-period MA
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > vol_ma_1h * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(vol_ma_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with volume confirmation and uptrend
            if (close[i] > r3_aligned[i] and vol_spike[i] and 
                close[i] > ema34_aligned[i] and volume[i] > vol_ma_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S3 with volume confirmation and downtrend
            elif (close[i] < s3_aligned[i] and vol_spike[i] and 
                  close[i] < ema34_aligned[i] and volume[i] > vol_ma_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to S3 (mean reversion to opposite level)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price returns to R3
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals