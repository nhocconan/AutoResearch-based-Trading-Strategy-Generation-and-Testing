#!/usr/bin/env python3
# Strategy: 4h_12h_Donchian20_Breakout_Volume_TrendFilter_v1
# Hypothesis: 4h Donchian(20) breakout with 12h EMA34 trend filter and volume confirmation. 
# Works in bull markets via breakout momentum and bear markets via trend-filtered short signals.
# Volume > 1.5x 20-period MA confirms institutional interest. ATR-based stop controls risk.
# Target: 20-40 trades/year to minimize fee drag. Uses 4h primary, 12h trend filter.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA34 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Load 4h data for entry signals, volume, ATR
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian channels (20-period on 4h)
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (20-period on 4h)
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility filter (14-period on 4h)
    high_low = high_4h - low_4h
    high_close = np.abs(high_4h - np.roll(close_4h, 1))
    low_close = np.abs(low_4h - np.roll(close_4h, 1))
    high_low[0] = high_4h[0] - low_4h[0]
    high_close[0] = np.abs(high_4h[0] - close_4h[0])
    low_close[0] = np.abs(low_4h[0] - close_4h[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper, above 12h EMA34 (uptrend), with volume confirmation
            if (price > highest_20[i] and 
                price > ema34_12h_aligned[i] and 
                vol > 1.5 * vol_ma_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, below 12h EMA34 (downtrend), with volume confirmation
            elif (price < lowest_20[i] and 
                  price < ema34_12h_aligned[i] and 
                  vol > 1.5 * vol_ma_20[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian lower or ATR-based stop
            if (price < lowest_20[i] or 
                price < high_4h[i] - 2.0 * atr_14[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian upper or ATR-based stop
            if (price > highest_20[i] or 
                price > low_4h[i] + 2.0 * atr_14[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_Donchian20_Breakout_Volume_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0