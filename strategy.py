#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + Chop Regime Filter
Hypothesis: Donchian(20) breakouts capture strong moves in both bull and bear markets.
Volume spike (>2x 20-period MA) confirms institutional participation.
Choppiness Index > 61.8 defines ranging markets where we avoid breakout trades to reduce false signals.
Designed for BTC/ETH with 75-200 total trades over 4 years to balance opportunity and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Chop Regime filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Chop Regime on 1d
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR(14)
    atr_14 = tr.rolling(window=14, min_periods=14).mean()
    
    # Highest High and Lowest Low over 14 periods
    hh_14 = high_1d.rolling(window=14, min_periods=14).max()
    ll_14 = low_1d.rolling(window=14, min_periods=14).min()
    
    # Chop = 100 * log10(sum(atr14) / (hh14 - ll14)) / log10(14)
    sum_atr = atr_14.rolling(window=14, min_periods=14).sum()
    chop = 100 * np.log10(sum_atr / (hh_14 - ll_14)) / np.log10(14)
    chop_values = chop.values
    
    # Align Chop to 4h timeframe
    chop_4h = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Calculate 20-period volume MA for volume spike confirmation (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate Donchian(20) channels (4h)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, volume MA, and Chop
    start_idx = max(20, 20)  # 20 for Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_ma = vol_ma_20[i]
        chop_val = chop_4h[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Chop regime: > 61.8 = ranging (avoid breakouts), < 38.2 = trending
        chop_ranging = chop_val > 61.8
        
        if position == 0:
            if chop_ranging:
                # Market ranging: avoid breakout trades
                signals[i] = 0.0
                position = 0
            else:
                # Trending market: look for Donchian breakouts with volume
                long_signal = (curr_close > upper) and volume_confirm
                short_signal = (curr_close < lower) and volume_confirm
                
                if long_signal:
                    signals[i] = 0.25
                    position = 1
                elif short_signal:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
                    position = 0
        elif position == 1:
            # Exit long: price closes below Donchian low OR chop becomes ranging
            if curr_close < lower or chop_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian high OR chop becomes ranging
            if curr_close > upper or chop_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0