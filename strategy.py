#!/usr/bin/env python3
# 4H_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike
# Hypothesis: Enter long when price breaks above Camarilla R3 level with 12h uptrend and volume spike;
# enter short when price breaks below S3 level with 12h downtrend and volume spike.
# Exit when price returns to Camarilla H4/L4 levels or trend reverses.
# Works in bull/bear by following 12h trend and using Camarilla levels for institutional support/resistance.
# Target: 20-40 trades/year per symbol.

name = "4H_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 12h data for Camarilla levels and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    # H4 = close + 1.1*(high - low)/1
    # L4 = close - 1.1*(high - low)/1
    rang = high_12h - low_12h
    r3 = close_12h + 1.1 * rang / 2
    s3 = close_12h - 1.1 * rang / 2
    h4 = close_12h + 1.1 * rang
    l4 = close_12h - 1.1 * rang
    
    # Trend: EMA50 on 12h close
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_12h = close_12h > ema50_12h
    downtrend_12h = close_12h < ema50_12h
    
    # Align all 12h indicators to 4h
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    h4_aligned = align_htf_to_ltf(prices, df_12h, h4)
    l4_aligned = align_htf_to_ltf(prices, df_12h, l4)
    uptrend_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h.astype(float))
    downtrend_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(h4_aligned[i]) or
            np.isnan(l4_aligned[i]) or np.isnan(uptrend_aligned[i]) or np.isnan(downtrend_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_spike = vol_ratio > 2.0
        
        is_uptrend = uptrend_aligned[i] > 0.5
        is_downtrend = downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: price breaks above R3 with 12h uptrend and volume spike
            if is_uptrend and volume_spike and close[i] > r3_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 with 12h downtrend and volume spike
            elif is_downtrend and volume_spike and close[i] < s3_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to H4 or trend turns down
            if close[i] < h4_aligned[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to L4 or trend turns up
            if close[i] > l4_aligned[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals