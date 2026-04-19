#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_KAMA_Trend_Filter_1d_Camarilla_Squeeze_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (trend direction)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # KAMA parameters: ER=10, fast=2, slow=30
    close_12h_series = pd.Series(close_12h)
    change = np.abs(close_12h_series.diff(10))
    volatility = close_12h_series.diff().abs().rolling(10, min_periods=10).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 1/30) + 1/30) ** 2
    kama = np.full_like(close_12h, np.nan)
    kama[9] = close_12h[9]  # seed
    for i in range(10, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # Get 1d data for Camarilla levels (entry levels)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range for Camarilla
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    range_1d = prev_high - prev_low
    
    # Camarilla levels: R3, S3 (stronger levels)
    r3 = prev_close + (range_1d * 1.1 / 4)  # R3 level
    s3 = prev_close - (range_1d * 1.1 / 4)  # S3 level
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        if np.isnan(kama_12h_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.8 * vol_ma
        
        # Trend filter: price above/below 12h KAMA
        price_above_kama = price > kama_12h_aligned[i]
        price_below_kama = price < kama_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume and uptrend
            if price > r3_aligned[i] and volume_ok and price_above_kama:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume and downtrend
            elif price < s3_aligned[i] and volume_ok and price_below_kama:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below S3 (mean reversion to opposite level)
            if price < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above R3 (mean reversion to opposite level)
            if price > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals