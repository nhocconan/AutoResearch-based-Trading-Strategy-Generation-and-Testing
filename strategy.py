#!/usr/bin/env python3
# 4H_CAMARILLA_VOLUME_SQUEEZE_V1
# Hypothesis: Camarilla pivot levels (from 1d) combined with Bollinger Band squeeze (volatility contraction)
# and volume confirmation provides high-probability breakout entries. Works in both bull and bear markets
# by capturing volatility expansion after contraction. Uses 4h timeframe with 1d Camarilla levels.
# Target: 20-40 trades/year to minimize fee drag while capturing strong moves.

name = "4H_CAMARILLA_VOLUME_SQUEEZE_V1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and BB squeeze
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Camarilla Pivot Levels (from 1d) ---
    # Based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r1 = close_1d + (range_1d * 1.1 / 12)
    r2 = close_1d + (range_1d * 1.1 / 6)
    r3 = close_1d + (range_1d * 1.1 / 4)
    r4 = close_1d + (range_1d * 1.1 / 2)
    
    s1 = close_1d - (range_1d * 1.1 / 12)
    s2 = close_1d - (range_1d * 1.1 / 6)
    s3 = close_1d - (range_1d * 1.1 / 4)
    s4 = close_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 4h
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    
    # --- Bollinger Band Squeeze (20, 2) on 1d ---
    # Measures volatility contraction - precursor to expansion
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean()
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Squeeze condition: BB width below 20-period average
    bb_width_ma = bb_width.rolling(window=20, min_periods=20).mean()
    squeeze = bb_width < bb_width_ma.values
    
    # Align squeeze to 4h
    squeeze_4h = align_htf_to_ltf(prices, df_1d, squeeze.values)
    
    # --- Volume Confirmation (4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if Camarilla levels or squeeze data is not ready
        if np.isnan(r1_4h[i]) or np.isnan(squeeze_4h[i]):
            if position != 0:
                # Hold position until clear exit
                signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Entry conditions: price near S3/R3 AND volatility squeeze AND volume spike
        # Long: price near S3 with bullish bias
        near_s3 = abs(close[i] - s3_4h[i]) / s3_4h[i] < 0.005  # Within 0.5% of S3
        # Short: price near R3 with bearish bias
        near_r3 = abs(close[i] - r3_4h[i]) / r3_4h[i] < 0.005  # Within 0.5% of R3
        
        long_entry = near_s3 and squeeze_4h[i] and vol_spike[i] and (close[i] > close[i-1])
        short_entry = near_r3 and squeeze_4h[i] and vol_spike[i] and (close[i] < close[i-1])
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price reaches opposite level OR squeeze releases
            if position == 1:
                # Exit if price reaches R3 or squeeze breaks down
                if close[i] >= r3_4h[i] or not squeeze_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit if price reaches S3 or squeeze breaks down
                if close[i] <= s3_4h[i] or not squeeze_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals