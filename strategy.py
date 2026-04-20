#!/usr/bin/env python3
# Strategy: 12h_1d_Donchian20_Breakout_Volume_Spike_TrendFilter_v1
# Hypothesis: Breakout above 20-period high or below 20-period low on 12h timeframe,
#             filtered by 1d EMA50 trend and volume > 2x 20-period MA. Designed for
#             20-40 trades/year to minimize fee drag and work in bull/bear markets.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Load 12h data for Donchian, volume, ATR
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Donchian channels (20-period on 12h)
    high_max_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (20-period on 12h)
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility filter (14-period on 12h)
    high_low = high_12h - low_12h
    high_close = np.abs(high_12h - np.roll(close_12h, 1))
    low_close = np.abs(low_12h - np.roll(close_12h, 1))
    high_low[0] = high_12h[0] - low_12h[0]
    high_close[0] = np.abs(high_12h[0] - close_12h[0])
    low_close[0] = np.abs(low_12h[0] - close_12h[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol = volume_12h[i]
        
        if position == 0:
            # Long: price breaks above 20-period high, above 1d EMA50 (uptrend), with volume confirmation
            if (price > high_max_20[i] and 
                price > ema50_1d_aligned[i] and 
                vol > 2.0 * vol_ma_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low, below 1d EMA50 (downtrend), with volume confirmation
            elif (price < low_min_20[i] and 
                  price < ema50_1d_aligned[i] and 
                  vol > 2.0 * vol_ma_20[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 20-period low or ATR-based stop
            if (price < low_min_20[i] or 
                price < high_12h[i] - 2.0 * atr_14[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 20-period high or ATR-based stop
            if (price > high_max_20[i] or 
                price > low_12h[i] + 2.0 * atr_14[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Donchian20_Breakout_Volume_Spike_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0