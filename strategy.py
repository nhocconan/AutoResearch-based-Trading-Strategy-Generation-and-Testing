#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + Volume Spike + ATR Stoploss + Chop Regime Filter
Hypothesis: Donchian breakouts capture institutional flow. Volume spike confirms
institutional participation. Chop regime filter (CHOP>61.8) ensures we only trade
in ranging markets where breakouts are more reliable. ATR stoploss manages risk.
Works in bull/bear via regime adaptation. Target: 25-40 trades/year on 4h.
"""

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
    
    # Calculate ATR(14) for stoploss and position sizing
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Calculate Donchian channels (20-period)
    if len(close) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # Calculate Choppiness Index (14-period) for regime filter
    if len(close) >= 14:
        # True Range
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_sum = tr.rolling(window=14, min_periods=14).sum().values
        
        # Max(HH) - Min(LL) over 14 periods
        max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        range_14 = max_high - min_low
        
        # Chop = 100 * log10(atr_sum / range_14) / log10(14)
        # Avoid division by zero
        chop = np.full(n, 50.0)  # default to neutral
        mask = (range_14 > 0) & (~np.isnan(range_14)) & (~np.isnan(atr_sum))
        chop[mask] = 100 * np.log10(atr_sum[mask] / range_14[mask]) / np.log10(14)
    else:
        chop = np.full(n, 50.0)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20[i] = np.mean(volume[:i+1]) if i >= 0 else 0.0
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        dh = donchian_high[i]
        dl = donchian_low[i]
        atr_val = atr[i]
        chop_val = chop[i]
        vol_spike = volume_spike[i]
        
        # Regime filter: only trade in ranging markets (Chop > 61.8)
        in_range = chop_val > 61.8
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume spike AND ranging market
            long_condition = (curr_high > dh) and vol_spike and in_range
            # Short: price breaks below Donchian low AND volume spike AND ranging market
            short_condition = (curr_low < dl) and vol_spike and in_range
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or Donchian low break
            if curr_close <= entry_price - 2.0 * atr_val or curr_low < dl:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or Donchian high break
            if curr_close >= entry_price + 2.0 * atr_val or curr_high > dh:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ChopFilter_ATRStop_v1"
timeframe = "4h"
leverage = 1.0