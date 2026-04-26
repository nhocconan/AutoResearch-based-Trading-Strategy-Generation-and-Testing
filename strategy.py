#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_ChopFilter_Volume
Hypothesis: Donchian(20) breakout on 4h with 1d EMA50 trend filter, choppiness regime (CHOP>61.8 = range), and volume confirmation. In trending markets (CHOP<=61.8), trade breakouts in direction of 1d EMA50. In ranging markets (CHOP>61.8), fade breaks at Donchian extremes with volume divergence. Discrete sizing (±0.25) targets ~30 trades/year to avoid fee drag. Works in bull/bear via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR(14)) / log10((HHV(14) - LLV(14)) * sqrt(14)))
    # Simplified: CHOP > 61.8 = range, CHOP < 38.2 = trend
    tr1 = pd.Series(df_1d['high']).ewm(span=14, adjust=False, min_periods=14).mean() - \
          pd.Series(df_1d['low']).ewm(span=14, adjust=False, min_periods=14).mean()
    tr1_abs = tr1.abs()
    atr14 = tr1_abs.ewm(span=14, adjust=False, min_periods=14).mean().values
    hh14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 * 14 / np.maximum(hh14 - ll14, 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian(20) on 4h
    donch_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of calculations (20 for Donchian/volume, 14 for chop/ATR, 50 for EMA)
    start_idx = max(20, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from calculation)
        if (np.isnan(donch_h[i]) or np.isnan(donch_l[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        donch_high = donch_h[i]
        donch_low = donch_l[i]
        ema_50_val = ema_50_1d_aligned[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        
        # Regime: CHOP > 61.8 = range, else trend
        is_range = chop_val > 61.8
        is_trend = chop_val <= 61.8
        
        # Determine 1d trend
        bullish_1d = close_val > ema_50_val
        bearish_1d = close_val < ema_50_val
        
        if is_trend:
            # Trending market: breakout in direction of 1d trend
            long_entry = (close_val > donch_high) and bullish_1d and vol_spike
            short_entry = (close_val < donch_low) and bearish_1d and vol_spike
            
            # Exit: trend reversal or opposite Donchian touch
            long_exit = not bullish_1d or (close_val < donch_low)
            short_exit = not bearish_1d or (close_val > donch_high)
        else:
            # Ranging market: fade at extremes (mean reversion)
            long_entry = (close_val < donch_low) and vol_spike  # Buy low
            short_entry = (close_val > donch_high) and vol_spike  # Sell high
            
            # Exit: return to mean (midpoint) or opposite extreme
            donch_mid = (donch_high + donch_low) / 2
            long_exit = close_val > donch_mid
            short_exit = close_val < donch_mid
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and long_exit:
            signals[i] = 0.0
            position = 0
        elif position == -1 and short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_ChopFilter_Volume"
timeframe = "4h"
leverage = 1.0