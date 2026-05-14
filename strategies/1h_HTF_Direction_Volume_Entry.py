#!/usr/bin/env python3
# 1h_HTF_Direction_Volume_Entry
# Hypothesis: Use 4h trend and 1d volatility filter for direction, 1h for precise entry.
# Long when 4h close > 4h EMA50 and 1d volatility low; enter on 1h pullback to EMA20 with volume.
# Short when 4h close < 4h EMA50 and 1d volatility low; enter on 1h bounce to EMA20 with volume.
# Designed for low trade frequency (15-30/year) to avoid fee drag, works in bull/bear via trend filter.

name = "1h_HTF_Direction_Volume_Entry"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMA50 trend
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_4h_up = close_4h > ema50_4h
    trend_4h_down = close_4h < ema50_4h
    
    # Align 4h trend to 1h
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up.astype(float))
    trend_4h_down_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_down.astype(float))
    
    # 1d ATR(20) for volatility filter (low volatility = ranging market good for mean reversion)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.absolute(high_1d[1:] - close_1d[:-1]), np.absolute(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[0], tr1])  # first value
    atr20_1d = pd.Series(tr1).rolling(window=20, min_periods=20).mean().values
    # Normalize ATR by price to get volatility percentage
    vol_pct = atr20_1d / close_1d
    # Low volatility threshold: bottom 30% of historical volatility
    vol_threshold = pd.Series(vol_pct).rolling(window=50, min_periods=50).quantile(0.3).values
    low_vol = vol_pct <= vol_threshold
    
    # Align 1d volatility filter to 1h
    low_vol_aligned = align_htf_to_ltf(prices, df_1d, low_vol.astype(float))
    
    # 1h EMA20 for entry timing
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: current volume > 1.5 * 20-period average (avoid low-volume noise)
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_4h_down_aligned[i]) or
            np.isnan(low_vol_aligned[i]) or np.isnan(ema20[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: 4h uptrend + low volatility + price near EMA20 with volume
            if (trend_4h_up_aligned[i] > 0.5 and 
                low_vol_aligned[i] > 0.5 and
                close[i] <= ema20[i] * 1.01 and  # within 1% above EMA20 (pullback)
                close[i] >= ema20[i] * 0.99 and  # within 1% below EMA20
                volume_filter):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + low volatility + price near EMA20 with volume
            elif (trend_4h_down_aligned[i] > 0.5 and 
                  low_vol_aligned[i] > 0.5 and
                  close[i] >= ema20[i] * 0.99 and  # within 1% below EMA20
                  close[i] <= ema20[i] * 1.01 and  # within 1% above EMA20
                  volume_filter):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: 4h trend fails or volatility increases
            if (trend_4h_up_aligned[i] < 0.5 or 
                low_vol_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: 4h trend fails or volatility increases
            if (trend_4h_down_aligned[i] < 0.5 or 
                low_vol_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals