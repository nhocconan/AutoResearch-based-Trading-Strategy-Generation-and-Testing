#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_Regime_ATRStop_v1
Hypothesis: On 4h timeframe, Donchian(20) breakout with volume confirmation (>2.0x 20-bar average volume) and choppiness regime filter (CHOP>61.8 for mean reversion, CHOP<38.2 for trend following) captures strong directional moves with reduced whipsaw. ATR-based stoploss (3.0x ATR) controls risk. Discrete sizing (0.25) minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for choppiness regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Choppiness Index (CHOP) for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(atr_1d)/log(hh_1d/ll_1d)) / log10(14)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    ratio = hh_1d / ll_1d
    # Avoid division by zero or log of zero/negative
    ratio = np.where(ratio > 1.0, ratio, 1.0)
    chop = 100 * (np.log10(sum_atr_14) / np.log10(ratio)) / np.log10(14)
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50.0, chop)  # neutral if invalid
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)
    
    # === 4h Donchian(20) channels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h volume confirmation (volume > 2.0x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma_20)
    
    # === 4h ATR for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 20  # max 5 days (20 * 4h = 80h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_confirmed[i]) or np.isnan(atr[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        chop_val = chop_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # Regime filters
        is_choppy = chop_val > 61.8  # range market -> mean revert
        is_trending = chop_val < 38.2  # trending market -> follow trend
        
        if position == 0:
            if is_trending:
                # Trending regime: breakout in direction of break
                long_condition = (price > donchian_high[i]) and vol_conf
                short_condition = (price < donchian_low[i]) and vol_conf
            elif is_choppy:
                # Choppy regime: mean reversion at channel edges
                long_condition = (price <= donchian_low[i]) and vol_conf
                short_condition = (price >= donchian_high[i]) and vol_conf
            else:
                # Neutral regime: no entries
                long_condition = False
                short_condition = False
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Check stoploss (3.0x ATR)
            if position == 1:
                if price < entry_price - 3.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 3.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_Regime_ATRStop_v1"
timeframe = "4h"
leverage = 1.0