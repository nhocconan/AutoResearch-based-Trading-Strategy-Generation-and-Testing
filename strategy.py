#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w volume confirmation and ATR volatility filter.
# Long when price breaks above 12h Donchian upper (20-period) AND 1w volume > 1.5x 20-week EMA AND ATR(14) < 50-day SMA(ATR).
# Short when price breaks below 12h Donchian lower (20-period) AND 1w volume > 1.5x 20-week EMA AND ATR(14) < 50-day SMA(ATR).
# Uses weekly volume for momentum confirmation and ATR-based volatility filter to avoid high-vol chop.
# Designed for low trade frequency (target: 15-25/year) to minimize fee drag and improve robustness.
# Works in bull markets via breakout follow-through and in bear markets via volatility-filtered breakdowns.
name = "12h_Donchian20_VolumeVolFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1w data for volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w volume > 1.5 * 20-week EMA
    vol_ema_20w = pd.Series(df_1w['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_filter_1w = np.where(vol_ema_20w > 0, df_1w['volume'].values / vol_ema_20w, 0.0) > 1.5
    vol_filter_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_filter_1w)
    
    # Load 1d data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # ATR(14) on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_list = []
    for i in range(len(close_1d)):
        if i == 0:
            tr = high_1d[0] - low_1d[0]
        else:
            tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        tr_list.append(tr)
    tr_1d = np.array(tr_list)
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 50-day SMA of ATR(14)
    atr_sma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # ATR(14) < 50-day SMA(ATR) = low volatility regime
    vol_low = atr_14 < atr_sma_50
    vol_low_aligned = align_htf_to_ltf(prices, df_1d, vol_low)
    
    # Load 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Donchian(20) on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_filter_1w_aligned[i]) or np.isnan(vol_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: break above Donchian high, weekly volume filter, low volatility
            long_condition = (close[i] > donchian_high_aligned[i]) and vol_filter_1w_aligned[i] and vol_low_aligned[i]
            # Short condition: break below Donchian low, weekly volume filter, low volatility
            short_condition = (close[i] < donchian_low_aligned[i]) and vol_filter_1w_aligned[i] and vol_low_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low or volatility increases
            if (close[i] < donchian_low_aligned[i]) or not vol_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high or volatility increases
            if (close[i] > donchian_high_aligned[i]) or not vol_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals