#!/usr/bin/env python3
# 1d_1w_donchian_volume_chop_v1
# Strategy: Daily Donchian(20) breakout with volume confirmation and weekly Chop filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture trend continuation. Volume confirms institutional interest.
# Chop filter avoids trades in choppy markets (Chop > 61.8). Works in bull/bear via trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Chop index (trend strength filter)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log15(sum(ATR14) / (maxHH - minLL))
    max_hh = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    denom = np.maximum(max_hh - min_ll, 1e-10)
    chop = 100 * np.log10(sum_atr / denom) / np.log10(15)
    
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)
    
    # Daily Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Chop regime filter: only trade when market is trending (Chop < 61.8)
        trending_market = chop_aligned[i] < 61.8
        
        # Entry logic: Donchian breakout + volume + Chop filter
        if (close[i] > donchian_high[i] and vol_confirm[i] and trending_market and position != 1):
            position = 1
            signals[i] = 0.25
        elif (close[i] < donchian_low[i] and vol_confirm[i] and trending_market and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: Donchian reversal or Chop > 61.8 (choppy)
        elif position == 1 and (close[i] < donchian_low[i] or chop_aligned[i] >= 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donchian_high[i] or chop_aligned[i] >= 61.8):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals