#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike
Hypothesis: Camarilla pivot levels (R3, S3) from daily timeframe with 1-day EMA34 trend filter and volume confirmation.
- Camarilla levels calculated from prior day's OHLC: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
- Long when price breaks above R3 with volume > 1.5x 20-period average and price > daily EMA34
- Short when price breaks below S3 with volume > 1.5x 20-period average and price < daily EMA34
- Exit when price returns to daily close level (pivot) or opposite Camarilla level is touched
- Designed to capture institutional breakouts with trend alignment in both bull and bear markets
- Target: 20-35 trades/year on 4h (80-140 total over 4 years)
"""

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
    
    # Daily data for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_pivot = close_1d  # Pivot point is previous day's close
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    pivot_4h = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i+1])
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for calculations
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or np.isnan(pivot_4h[i]) or 
            np.isnan(ema34_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with volume confirmation and uptrend
            if (close[i] > r3_4h[i] and volume_filter[i] and close[i] > ema34_4h[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with volume confirmation and downtrend
            elif (close[i] < s3_4h[i] and volume_filter[i] and close[i] < ema34_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to pivot or touches S3 (contrarian level)
            if (close[i] <= pivot_4h[i] or close[i] < s3_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to pivot or touches R3 (contrarian level)
            if (close[i] >= pivot_4h[i] or close[i] > r3_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0