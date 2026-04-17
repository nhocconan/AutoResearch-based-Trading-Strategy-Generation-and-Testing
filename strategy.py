#!/usr/bin/env python3
"""
1h_4h1d_Squeeze_Breakout_Volume
Hypothesis: Bollinger Band squeeze (BB width < 20th percentile) on 1h indicates low volatility.
Breakout from squeeze with volume > 1.5x 20-period average and aligned with 4h trend (EMA50) 
and 1d trend (price > 200 EMA). Designed for low frequency (15-37/year) to catch explosive 
moves after consolidation in both bull and bear markets. Uses 1h for entry timing, 4h/1d for 
trend filter and volatility regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Bollinger Bands for squeeze detection ===
    bb_length = 20
    bb_mult = 2.0
    basis = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    dev = bb_mult * pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    upper = basis + dev
    lower = basis - dev
    bb_width = upper - lower
    
    # Squeeze condition: BB width < 20th percentile of last 50 periods
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=10).quantile(0.20).values
    squeeze = bb_width < bb_width_percentile
    
    # === 4h EMA50 for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d EMA200 for long-term trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === Volume confirmation: current volume > 1.5x 20-period average ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Session filter: 08:00-20:00 UTC ===
    hours = prices.index.hour  # pre-computed DatetimeIndex.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for BB (20), percentile (50), 4h EMA (50), 1d EMA (200)
    warmup = 200
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(bb_width[i]) or np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Session filter: only trade 08:00-20:00 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry conditions (only when flat)
        if position == 0:
            # Squeeze breakout conditions
            bb_squeeze = squeeze[i]
            vol_filter = volume[i] > 1.5 * vol_ma_20[i]
            price_above_bb = close[i] > upper[i]
            price_below_bb = close[i] < lower[i]
            
            # Trend filters: 4h EMA50 and 1d EMA200
            uptrend_4h = close[i] > ema_50_4h_aligned[i]
            uptrend_1d = close[i] > ema_200_1d_aligned[i]
            downtrend_4h = close[i] < ema_50_4h_aligned[i]
            downtrend_1d = close[i] < ema_200_1d_aligned[i]
            
            # Long: squeeze breakout up + volume + uptrend on both timeframes
            if bb_squeeze and vol_filter and price_above_bb and uptrend_4h and uptrend_1d:
                signals[i] = 0.20
                position = 1
                continue
            # Short: squeeze breakout down + volume + downtrend on both timeframes
            elif bb_squeeze and vol_filter and price_below_bb and downtrend_4h and downtrend_1d:
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit conditions
        elif position == 1:
            # Exit long: price closes below 4h EMA50 or 1d EMA200
            if close[i] < ema_50_4h_aligned[i] or close[i] < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price closes above 4h EMA50 or 1d EMA200
            if close[i] > ema_50_4h_aligned[i] or close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_Squeeze_Breakout_Volume"
timeframe = "1h"
leverage = 1.0