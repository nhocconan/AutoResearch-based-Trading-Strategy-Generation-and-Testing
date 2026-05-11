#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: On 6h timeframe, price breaking above Camarilla R3 or below S3 with 1d trend alignment (close > 1d EMA34) and volume surge (volume > 1.5x 20-period average) indicates strong momentum continuation. Inverse for shorts. Uses daily timeframe for trend and volume confirmation to filter false breaks, targeting 20-50 trades per year per symbol. Works in bull/bear via trend filter.
"""

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla calculation, trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Handle first bar
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    # Camarilla levels
    range_ = prev_high - prev_low
    camarilla_mult = 1.1 / 12  # ≈ 0.091666
    r3 = prev_close + range_ * camarilla_mult * 4
    s3 = prev_close - range_ * camarilla_mult * 4
    
    # Align Camarilla levels to 6h
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA34 for trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 6h price and volume
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    signals = np.zeros(n)
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 1d average volume (scaled)
        # Scale 1d volume to 6h: 1d has 4x 6h bars, so divide by 4
        vol_threshold = vol_avg_1d_aligned[i] / 4.0 * 1.5
        vol_confirm = volume_6h[i] > vol_threshold
        
        if vol_confirm:
            # Long: price > R3 and above 1d EMA34 (uptrend)
            if close_6h[i] > r3_6h[i] and close_6h[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
            # Short: price < S3 and below 1d EMA34 (downtrend)
            elif close_6h[i] < s3_6h[i] and close_6h[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
    
    return signals