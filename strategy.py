#!/usr/bin/env python3
# 6h_1d_1w_TripleTimeframeConfluence
# Hypothesis: A confluence of trend across 1d (EMA34), 1w (EMA34), and 6h (price above/below EMA34)
# creates high-probability directional moves. Entry occurs when all three timeframes align
# and price breaks the 6h Donchian channel (20) with volume confirmation (>1.5x 20-period average).
# Exit when the 6d trend weakens (price crosses below/above 6h EMA34) or volatility drops.
# Designed for low frequency (15-30 trades/year) to minimize fee drag while capturing strong trends.
# Works in bull markets (aligned uptrends) and bear markets (aligned downtrends).

name = "6h_1d_1w_TripleTimeframeConfluence"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d and 1w data for trend filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 34 or len(df_1w) < 34:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d EMA34 for trend filter ---
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- 1w EMA34 for trend filter ---
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # --- 6h EMA34 for exit signal ---
    ema_34_6h = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # --- 6h Donchian channel (20) for breakout signals ---
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # --- Volume confirmation (1.5x 20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for all indicators
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(ema_34_6h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: all timeframes bullish + break above Donchian high + volume surge
            if (ema_34_1d_aligned[i] < close[i] and      # 1d uptrend
                ema_34_1w_aligned[i] < close[i] and      # 1w uptrend
                close[i] > donchian_high[i] and          # breakout above 6h Donchian high
                volume_surge):
                signals[i] = 0.25
                position = 1
            # Short: all timeframes bearish + break below Donchian low + volume surge
            elif (ema_34_1d_aligned[i] > close[i] and    # 1d downtrend
                  ema_34_1w_aligned[i] > close[i] and    # 1w downtrend
                  close[i] < donchian_low[i] and         # breakdown below 6h Donchian low
                  volume_surge):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses below 6h EMA34 OR trend misalignment
                if close[i] < ema_34_6h[i] or ema_34_1d_aligned[i] > close[i] or ema_34_1w_aligned[i] > close[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above 6h EMA34 OR trend misalignment
                if close[i] > ema_34_6h[i] or ema_34_1d_aligned[i] < close[i] or ema_34_1w_aligned[i] < close[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals