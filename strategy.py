#!/usr/bin/env python3
"""
12h_ChoppinessIndex_VolumeSpike_Breakout_V1
Hypothesis: In low-chop regimes (trending markets), price breakouts from the prior 12h candle with volume spike capture strong moves in both bull and bear markets. Uses 1d timeframe for chop regime filter and ATR-based trailing stop. Target: 12-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Choppiness Index (14-period) on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum_atr_14 / (hh_14 - ll_14)) / log10(14)
    range_14 = hh_14 - ll_14
    chop = np.where(range_14 > 0, 100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 100)
    chop = np.where(np.isnan(chop), 100, chop)
    
    # Align chop to 12h timeframe (no extra delay needed for chop)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h price channel: prior candle high/low
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Prior 12h candle high/low (shifted by 1)
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    # Volume filter: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (12h timeframe)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(prior_high[i]) or np.isnan(prior_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Regime filter: chop < 38.2 = trending (favor breakouts)
        trending_regime = chop_aligned[i] < 38.2
        
        # Volume confirmation
        volume_ok = volume > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: break above prior high in trending regime with volume
            if trending_regime and volume_ok:
                if price > prior_high[i]:
                    signals[i] = 0.30
                    position = 1
            # Short: break below prior low in trending regime with volume
            elif trending_regime and volume_ok:
                if price < prior_low[i]:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:
            # Exit: price reaches prior low or ATR stoploss
            if price <= prior_low[i] or price < np.maximum.accumulate(close[:i+1])[-1] - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit: price reaches prior high or ATR stoploss
            if price >= prior_high[i] or price < np.minimum.accumulate(close[:i+1])[-1] + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_ChoppinessIndex_VolumeSpike_Breakout_V1"
timeframe = "12h"
leverage = 1.0