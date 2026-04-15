# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 1w ADX trend filter
# Donchian breakout captures trend continuation; volume spike confirms institutional interest;
# Weekly ADX > 25 ensures we only trade in strong trends, avoiding whipsaws in ranging markets.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue).
# Designed for low trade frequency (~20-40/year) with clear trend-following logic.

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
    volume_1d = df_1d['volume'].values
    
    # Load 1w data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1d volume spike: volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * vol_ma)
    
    # 1w ADX(14) for trend strength
    tr1 = np.maximum(high_1w[1:], low_1w[:-1]) - np.minimum(high_1w[1:], low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1w + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1w + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1w = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align indicators to 4h timeframe
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_spike_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            continue
        
        # Only trade when weekly ADX indicates strong trend (ADX > 25)
        if adx_1w_aligned[i] > 25:
            # Long breakout: price breaks above 20-period high with volume spike
            if close[i] > highest_high[i] and vol_spike_aligned[i] > 0.5 and position <= 0:
                position = 1
                signals[i] = position_size
            # Short breakdown: price breaks below 20-period low with volume spike
            elif close[i] < lowest_low[i] and vol_spike_aligned[i] > 0.5 and position >= 0:
                position = -1
                signals[i] = -position_size
            # Exit when price returns to the middle of the channel
            elif position == 1 and close[i] < (highest_high[i] + lowest_low[i]) / 2:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > (highest_high[i] + lowest_low[i]) / 2:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1dVolSpike_1wADXTrend"
timeframe = "4h"
leverage = 1.0