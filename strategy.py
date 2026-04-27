#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_Regime_CombinedFilter
Hypothesis: Donchian(20) breakout with volume confirmation and regime filter (choppiness/ADX) to avoid whipsaw in ranging markets.
Uses discrete position sizing (0.25) and exits on trend reversal (close crosses opposite Donchian band).
Designed to work in both bull and bear markets by filtering breakouts with volume and regime conditions.
"""

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
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Choppiness regime filter: CHOP > 61.8 = ranging (avoid breakouts in ranging markets)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values / (max_high_14 - min_low_14)) / np.log10(14)
    chop_regime = chop > 61.8  # True = ranging market (avoid breakouts)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # Discrete size to reduce fee churn
    
    # Warmup: need Donchian(20), EMA34, vol avg, chop
    start_idx = max(34, 20, 20, 28)  # EMA34 + Donchian20 + vol20 + chop28
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_band = highest_20[i]
        lower_band = lowest_20[i]
        ema_val = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        is_ranging = chop_regime[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with EMA alignment, volume spike, and NOT in ranging market
            long_condition = (close_val > upper_band and 
                            close_val > ema_val and 
                            vol_spike and 
                            not is_ranging)
            short_condition = (close_val < lower_band and 
                             close_val < ema_val and 
                             vol_spike and 
                             not is_ranging)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price crosses below lower Donchian band (trend reversal)
            if close_val < lower_band:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above upper Donchian band (trend reversal)
            if close_val > upper_band:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_Volume_Regime_CombinedFilter"
timeframe = "4h"
leverage = 1.0