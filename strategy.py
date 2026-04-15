#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d ADX trend filter
# Donchian breakout captures momentum in trending markets
# Volume confirmation ensures breakout is genuine (not fakeout)
# 1d ADX > 25 filters for trending regime, avoiding false signals in ranging markets
# Designed for low trade frequency (target 20-40/year) with clear trend following logic
# Works in bull markets (breakouts up) and bear markets (breakouts down)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 4h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # 1d ADX(14) for trend strength
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(high_1d[1:], low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(adx_aligned[i])):
            continue
        
        # Only trade in trending markets (ADX > 25)
        if adx_aligned[i] > 25:
            # Long breakout: price breaks above Donchian high with volume confirmation
            if close[i] > donchian_high[i] and volume_confirm[i] and position <= 0:
                position = 1
                signals[i] = position_size
            # Short breakdown: price breaks below Donchian low with volume confirmation
            elif close[i] < donchian_low[i] and volume_confirm[i] and position >= 0:
                position = -1
                signals[i] = -position_size
            # Exit when price returns to middle of channel (mean reversion within trend)
            elif position == 1 and close[i] < (donchian_high[i] + donchian_low[i]) / 2:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > (donchian_high[i] + donchian_low[i]) / 2:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Volume_ADXTrend"
timeframe = "4h"
leverage = 1.0