#!/usr/bin/env python3
"""
12h_Donchian_20_Breakout_Trend_Volume_Regime
Hypothesis: Donchian(20) breakouts on 12h with volume confirmation, 1d trend filter (EMA50), and chop regime filter (avoid range) yields low-frequency, high-quality trades. Designed for 12-37 trades/year to minimize fee drag in choppy markets like 2025.
"""

name = "12h_Donchian_20_Breakout_Trend_Volume_Regime"
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
    
    # Get 1d data for trend and chop filters (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d trend filter: EMA(50) on close
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Chop regime filter: Chop(14) < 61.8 = trending (we want trend, not range)
    # Chop = 100 * log10(sum(ATR(1), n) / (max(high, n) - min(low, n))) / log10(n)
    atr_1d = np.abs(df_1d['high'] - df_1d['low']).values
    tr14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(tr14 / (max_high - min_low)) / np.log10(14)
    chop = np.where((max_high - min_low) == 0, 100, chop)  # avoid div by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels on 12h: 20-period high/low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above Donchian high, volume confirmation, 1d uptrend, trending regime
            if (close[i] > donchian_high[i] and 
                volume_filter[i] and 
                close[i] > ema50_1d_aligned[i] and 
                chop_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low, volume confirmation, 1d downtrend, trending regime
            elif (close[i] < donchian_low[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  chop_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Donchian low (failed breakout) OR trend breaks
            if (close[i] < donchian_low[i] or 
                close[i] < ema50_1d_aligned[i] or 
                chop_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Donchian high (failed breakdown) OR trend breaks
            if (close[i] > donchian_high[i] or 
                close[i] > ema50_1d_aligned[i] or 
                chop_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals