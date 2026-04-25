#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ATRStop_RegimeFilter
Hypothesis: On 4h timeframe, Donchian(20) breakouts with volume spike (>2.0x 20-bar avg) and chop regime filter (CHOP>61.8) capture institutional breakouts in ranging markets while avoiding whipsaws in strong trends. Uses ATR-based stoploss (2.5*ATR) for risk control. Designed for 20-50 trades/year to minimize fee drag. Works in bull markets via long breakouts and bear markets via short breakouts. Uses discrete position sizing (0.30) to reduce churn. Primary timeframe: 4h, HTF: 1d for trend filter.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d for trend filter (long-term bias)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss and position sizing
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(highest_high - lowest_low)) / log10(14)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_ll = highest_high - lowest_low
    chop = 100 * np.log10(atr_sum) / np.log10(hh_ll) / np.log10(14)
    chop = np.where((hh_ll > 0) & (atr_sum > 0), chop, 50)  # default to neutral when invalid
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 14)  # EMA50, Donchian, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Get aligned values
        ema_val = ema_50_aligned[i]
        highest_val = highest_high[i]
        lowest_val = lowest_low[i]
        atr_val = atr[i]
        chop_val = chop[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        in_range = chop_val > 61.8
        
        if position == 0:
            # Look for entry signals: Donchian breakout with volume spike and range regime
            # Long: price breaks above Donchian high with volume spike and in range
            long_signal = (high_val > highest_val) and volume_spike and in_range
            # Short: price breaks below Donchian low with volume spike and in range
            short_signal = (low_val < lowest_val) and volume_spike and in_range
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
                highest_since_entry = high_val
                lowest_since_entry = low_val
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
                highest_since_entry = high_val
                lowest_since_entry = low_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high_val)
            
            # Exit conditions:
            # 1. ATR-based stoploss: price drops 2.5*ATR from entry
            if close_val < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Donchian breakout in opposite direction: price breaks below Donchian low
            elif low_val < lowest_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low_val)
            
            # Exit conditions:
            # 1. ATR-based stoploss: price rises 2.5*ATR from entry
            if close_val > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Donchian breakout in opposite direction: price breaks above Donchian high
            elif high_val > highest_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ATRStop_RegimeFilter"
timeframe = "4h"
leverage = 1.0