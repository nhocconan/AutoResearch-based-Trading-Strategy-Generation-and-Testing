#!/usr/bin/env python3
"""
6h_12h_1d_WickReversal_VolumeSpike_v1
Concept: 6h rejection candles (wick > 2x body) at 12h/1d support/resistance with volume spike.
- Long: Close > Open (bullish) AND (Open - Low) > 2*(Close - Open) AND Close > 12h EMA(50) AND 1d volume > 2x 20-period avg
- Short: Close < Open (bearish) AND (High - Close) > 2*(Open - Close) AND Close < 12h EMA(50) AND 1d volume > 2x 20-period avg
- Exit: Opposite wick signal OR loss of 12h EMA(50) filter
- Position sizing: 0.25
- Works in bull/bear: volume confirms institutional interest, EMA filter adapts to trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_WickReversal_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 12h: EMA Trend Filter (50) ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 1d: Volume MA (20-period) ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 1d: Current Volume ===
    volume_1d_vals = df_1d['volume'].values
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_vals)
    
    # === 6h: Price Action ===
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate body and wicks
    body = np.abs(close - open_)
    lower_wick = np.where(close >= open_, open_ - low, close - low)
    upper_wick = np.where(close >= open_, high - close, high - open_)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        ema_50 = ema_50_12h_aligned[i]
        vol_ma_20 = vol_ma_20_1d_aligned[i]
        vol_1d = volume_1d_aligned[i]
        o = open_[i]
        h = high[i]
        l = low[i]
        c = close[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_50) or np.isnan(vol_ma_20) or np.isnan(vol_1d) or 
            np.isnan(o) or np.isnan(h) or np.isnan(l) or np.isnan(c)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current daily volume > 2.0x 20-period average
        vol_condition = vol_1d > 2.0 * vol_ma_20
        
        # Wick conditions
        body_size = np.abs(c - o)
        # Avoid division by zero - if body is too small, treat as doji
        if body_size < 1e-8:
            is_bullish_wick = False
            is_bearish_wick = False
        else:
            # Bullish rejection: long lower wick, small body, close near high
            is_bullish_wick = (c >= o) and (lower_wick[i] > 2.0 * body_size)
            # Bearish rejection: long upper wick, small body, close near low
            is_bearish_wick = (c <= o) and (upper_wick[i] > 2.0 * body_size)
        
        if position == 0:
            # Long: bullish rejection at support with volume spike and above 12h EMA
            if is_bullish_wick and c > ema_50 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: bearish rejection at resistance with volume spike and below 12h EMA
            elif is_bearish_wick and c < ema_50 and vol_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish rejection OR price breaks below 12h EMA
            if is_bearish_wick or c < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish rejection OR price breaks above 12h EMA
            if is_bullish_wick or c > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals