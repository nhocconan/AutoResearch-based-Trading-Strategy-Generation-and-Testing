#!/usr/bin/env python3
# 12H_DONCHIAN_BREAKOUT_1D_TREND_FILTER
# Hypothesis: Use daily Donchian channels for trend direction and 12h Donchian breakouts for entry.
# Enter long when 12h price breaks above daily Donchian high with volume confirmation in bullish regime,
# enter short when price breaks below daily Donchian low with volume confirmation in bearish regime.
# Use 12h ATR for stoploss. Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

name = "12H_DONCHIAN_BREAKOUT_1D_TREND_FILTER"
timeframe = "12h"
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
    
    # Daily data for trend filter (Donchian channels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily Donchian channels (20-period)
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA for trend filter (34-period)
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 12h Donchian breakout levels (20-period)
    donchian_high_12h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_12h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h ATR for stoploss and volatility filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike detection (20-period volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > vol_ma * 1.5  # Volume at least 1.5x average
    
    # Align daily indicators to 12h timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(donchian_high_12h[i]) or 
            np.isnan(donchian_low_12h[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 12h price breaks above daily Donchian high with volume confirmation in bullish regime
            if (close[i] > donchian_high_20_aligned[i] and vol_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 12h price breaks below daily Donchian low with volume confirmation in bearish regime
            elif (close[i] < donchian_low_20_aligned[i] and vol_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to daily Donchian mean or stoploss
            if close[i] < (donchian_high_20_aligned[i] + donchian_low_20_aligned[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to daily Donchian mean or stoploss
            if close[i] > (donchian_high_20_aligned[i] + donchian_low_20_aligned[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals