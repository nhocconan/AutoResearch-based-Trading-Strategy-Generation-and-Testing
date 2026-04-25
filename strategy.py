#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d ATR Trend + Volume Spike
Hypothesis: Donchian channel breakouts capture institutional flow. 1d ATR trend filter adapts to volatility regimes.
Volume spike confirms conviction. Works in bull/bear via ATR-based trend. Target: 20-50 trades/year on 4h.
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
    
    # Get 1d data for ATR trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for trend filter (higher ATR = stronger trend)
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate ATR(14) for stoploss on 4h
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for Donchian(20) and ATR
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        atr_1d = atr_14_1d_aligned[i]
        atr_val = atr[i]
        
        # Donchian(20) channels
        if i >= 20:
            donchian_high = np.max(high[i-19:i+1])
            donchian_low = np.min(low[i-19:i+1])
        else:
            donchian_high = np.max(high[:i+1])
            donchian_low = np.min(low[:i+1])
        
        # Volume spike: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 1.5 * vol_ma_20
        
        # ATR-based trend filter: trend strength = current ATR vs 20-period average
        if i >= 20:
            atr_ma_20 = np.mean(atr[i-19:i+1])
        else:
            atr_ma_20 = np.mean(atr[:i+1])
        # Strong trend when current ATR > 1.2 * average ATR
        strong_trend = atr_val > 1.2 * atr_ma_20
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume spike AND strong trend
            long_condition = (curr_high > donchian_high) and volume_spike and strong_trend
            # Short: price breaks below Donchian low AND volume spike AND strong trend
            short_condition = (curr_low < donchian_low) and volume_spike and strong_trend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
        elif position == 1:
            # Update highest since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit long: stoploss (2.5*ATR below highest) or weak trend
            if curr_close <= highest_since_entry - 2.5 * atr_val or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit short: stoploss (2.5*ATR above lowest) or weak trend
            if curr_close >= lowest_since_entry + 2.5 * atr_val or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATR_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0