#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_Regime_CombinedFilter
Hypothesis: 4h strategy using Donchian(20) breakout with volume confirmation and choppiness regime filter. 
In trending markets (CHOP < 38.2), trade breakouts in direction of trend. 
In ranging markets (CHOP > 61.8), fade breakouts at extremes. 
Volume spike confirms institutional participation. 
Designed for BTC/ETH robustness via regime adaptation. 
Target 75-200 trades over 4 years (19-50/year) with position size 0.25.
Uses discrete levels to minimize fee drag.
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
    
    # Get 12h data for choppiness regime filter
    df_12h = get_htf_data(prices, '12h')
    # Calculate choppiness index: CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    # Simplified: CHOP = 100 * log10(ATR_sum / (max_high - min_low)) / log10(period)
    # We'll use a proxy: CHOP = 100 * (ATR(14) / (HHV(14) - LLV(14))) normalized
    # For simplicity, use: CHOP = 100 * (ATR(14) / (max(high) - min(low) over 12h)) 
    # But we need rolling. Instead, use standard formula approximation:
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).mean().values
    hh14 = df_12h['high'].rolling(window=14, min_periods=14).max().values
    ll14 = df_12h['low'].rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 * 14 / (hh14 - ll14 + 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop.values)
    
    # Donchian channels (20-period)
    donch_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Fixed position size to minimize churn
    
    # Warmup: need Donchian(20), vol avg(20), 12h CHOP(14) -> max(20,20,14+12*? but aligned)
    start_idx = 20  # Donchian and vol need 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_h[i]) or np.isnan(donch_l[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = donch_h[i]
        lower = donch_l[i]
        chop_val = chop_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine regime
            if chop_val < 38.2:  # Trending regime
                # Trade breakouts in direction of momentum (use price action)
                # Simple momentum: close > open for bullish bias
                momentum_bias = 1 if close[i] > prices['open'].iloc[i] else -1
                long_condition = (close_val > upper and 
                                momentum_bias > 0 and 
                                vol_conf)
                short_condition = (close_val < lower and 
                                 momentum_bias < 0 and 
                                 vol_conf)
            elif chop_val > 61.8:  # Ranging regime
                # Fade breakouts at extremes (mean reversion)
                long_condition = (close_val < lower and  # Oversold bounce
                                vol_conf)
                short_condition = (close_val > upper and   # Overbought rejection
                                 vol_conf)
            else:  # Choppy regime (38.2-61.8) - no clear edge
                long_condition = False
                short_condition = False
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price retreats to midpoint or opposite band touch
            midpoint = (upper + lower) / 2
            if close_val < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises to midpoint or opposite band touch
            midpoint = (upper + lower) / 2
            if close_val > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_Volume_Regime_CombinedFilter"
timeframe = "4h"
leverage = 1.0