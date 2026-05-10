#!/usr/bin/env python3
# 12h_1W_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: Weekly trend filter (EMA20) reduces false breakouts in choppy markets,
# while weekly Camarilla R3/S3 levels provide precise entries on 12h timeframe.
# Volume confirmation ensures breakout strength. Designed for low trade frequency (12-37/year) to minimize fee drag.
# Works in bull markets via trend-following breakouts and in bear via mean-reversion
# at extreme levels when trend aligns.

name = "12h_1W_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "12h"
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
    
    # Get weekly data for trend filter and Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA20 for trend (more stable than SMA)
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate weekly typical price and range from previous week for Camarilla
    typical_price = (high_1w + low_1w + close_1w) / 3
    range_hl = high_1w - low_1w
    # Camarilla R3 and S3 levels
    R3 = typical_price + (range_hl * 1.2500)
    S3 = typical_price - (range_hl * 1.2500)
    # Align weekly levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3.values)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3.values)
    
    # Volume confirmation (20-period average on 12h = ~10 days)
    vol_ma_period = 20
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, vol_ma_period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20) + 5  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or \
           np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0x average (stricter for fewer trades)
        volume_confirm = volume[i] > 2.0 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above R3 with volume, above weekly EMA20 (uptrend)
            if close[i] > R3_aligned[i] and volume_confirm and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume, below weekly EMA20 (downtrend)
            elif close[i] < S3_aligned[i] and volume_confirm and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S3 or breaks below weekly EMA20
            if close[i] < S3_aligned[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above R3 or breaks above weekly EMA20
            if close[i] > R3_aligned[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals