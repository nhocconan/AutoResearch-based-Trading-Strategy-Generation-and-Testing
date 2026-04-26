#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Trade Camarilla R3/S3 breakouts on 6h with 1-week EMA50 trend filter and volume spike (1.8x median) only. Designed for ~80 trades/year to balance edge and fee drag. Weekly trend filter provides robustness in bull/bear markets by avoiding counter-trend trades. Uses ATR(14) trailing stop (2.0x ATR) for risk control. Focus on BTC/ETH as primary symbols.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 1w OHLC
    prev_close_1w = df_1w['close'].shift(1).values
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values
    
    camarilla_r3 = prev_close_1w + 3.000/8 * (prev_high_1w - prev_low_1w)
    camarilla_s3 = prev_close_1w - 3.000/8 * (prev_high_1w - prev_low_1w)
    camarilla_r4 = prev_close_1w + 1.000/2 * (prev_high_1w - prev_low_1w)
    camarilla_s4 = prev_close_1w - 1.000/2 * (prev_high_1w - prev_low_1w)
    
    # Align HTF indicators to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Volume confirmation: 1.8x median volume (24-period for 6h = ~4 days)
    vol_median = pd.Series(volume).rolling(window=24, min_periods=24).median().values
    
    # ATR(14) for volatility-based stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA(50) 1w, volume median (24), ATR (14)
    start_idx = max(50, 24, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_median[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_1w_val
        downtrend = close_val < ema_50_1w_val
        
        if position == 0:
            # Long: break above R3 with volume spike, and uptrend
            long_signal = (close_val > camarilla_r3_aligned[i]) and \
                          (volume_val > 1.8 * vol_median_val) and \
                          uptrend
            
            # Short: break below S3 with volume spike, and downtrend
            short_signal = (close_val < camarilla_s3_aligned[i]) and \
                           (volume_val > 1.8 * vol_median_val) and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop
            if close_val < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop
            if close_val > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0