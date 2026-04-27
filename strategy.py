#!/usr/bin/env python3
"""
6h Volume-Weighted VWAP Deviation with 12h Trend Filter and Volume Spike.
Long when price > VWAP(20) AND price > 12h EMA50 AND volume > 2x average.
Short when price < VWAP(20) AND price < 12h EMA50 AND volume > 2x average.
Exit when price crosses back across VWAP.
Designed to capture institutional flow with volume confirmation in both bull and bear markets.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # VWAP(20) calculation
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.zeros(n, dtype=np.float64)
    vwap_denominator = np.zeros(n, dtype=np.float64)
    vwap = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(n):
        vwap_numerator[i] = typical_price[i] * volume[i]
        vwap_denominator[i] = volume[i]
        
        if i >= 19:
            num_sum = np.sum(vwap_numerator[i-19:i+1])
            den_sum = np.sum(vwap_denominator[i-19:i+1])
            if den_sum > 0:
                vwap[i] = num_sum / den_sum
    
    # Volume filter: volume > 2x 20-period average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need VWAP(20) + EMA50_12h + volume MA
    start_idx = max(19, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vwap[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price_now = close[i]
        vwap_val = vwap[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 2x average
        vol_filter = vol_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: price > VWAP AND price > 12h EMA50 AND volume spike
            if price_now > vwap_val and price_now > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price < VWAP AND price < 12h EMA50 AND volume spike
            elif price_now < vwap_val and price_now < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below VWAP
            if price_now < vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above VWAP
            if price_now > vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Volume_Weighted_VWAP_Deviation_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0