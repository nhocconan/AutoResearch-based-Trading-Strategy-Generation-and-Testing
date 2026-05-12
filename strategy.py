#!/usr/bin/env python3
name = "4h_Donchian20_TrendVolume_Regime"
timeframe = "4h"
leverage = 1.0

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
    
    # Daily trend filter: EMA(34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Weekly regime filter: Choppiness Index (14)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(tr)/ (hh - ll)) / log10(14)
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    denominator = hh_1w - ll_1w
    chop = np.where(denominator > 0, 100 * np.log10(sum_tr / denominator) / np.log10(14), 50)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Donchian channels (20-period) on 4h
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure all indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend condition: price above/below daily EMA(34)
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Regime condition: chop < 38.2 (trending market)
        trending_regime = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long: Donchian breakout up + trend up + trending regime + volume
            if (close[i] > donch_high[i] and trend_up and 
                trending_regime and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + trend down + trending regime + volume
            elif (close[i] < donch_low[i] and trend_down and 
                  trending_regime and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Donchian break down OR trend change
            if close[i] < donch_low[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Donchian break up OR trend change
            if close[i] > donch_high[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals