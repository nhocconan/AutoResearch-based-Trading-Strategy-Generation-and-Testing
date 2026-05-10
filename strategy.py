#!/usr/bin/env python3
# 1h_Camarilla_R3_S3_Breakout_4hTrend_Volume
# Hypothesis: Use 4h EMA50 for trend direction (reduces false breakouts), hourly Camarilla R3/S3 for precise entries, and volume confirmation to ensure breakout strength.
# Designed for low trade frequency (15-37/year) to minimize fee drag on 1h timeframe.
# Works in bull markets via trend-following breakouts and in bear via mean-reversion at extreme levels when trend aligns.

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # 4h EMA50 for trend (more stable than SMA)
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    # Calculate typical price and range from previous day
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    # Camarilla R3 and S3 levels
    R3 = typical_price + (range_hl * 1.2500)
    S3 = typical_price - (range_hl * 1.2500)
    # Align daily levels to 1h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3.values)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3.values)
    
    # Volume confirmation (24-period average on 1h = 1 day)
    vol_ma_period = 24
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, vol_ma_period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 50) + 5  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or \
           np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0x average (stricter for fewer trades)
        volume_confirm = volume[i] > 2.0 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above R3 with volume, above 4h EMA50 (uptrend)
            if close[i] > R3_aligned[i] and volume_confirm and close[i] > ema_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 with volume, below 4h EMA50 (downtrend)
            elif close[i] < S3_aligned[i] and volume_confirm and close[i] < ema_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price closes below S3 or breaks below 4h EMA50
            if close[i] < S3_aligned[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price closes above R3 or breaks above 4h EMA50
            if close[i] > R3_aligned[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals